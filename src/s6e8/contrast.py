from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def rank01(values: np.ndarray) -> np.ndarray:
    """Return deterministic average percentile ranks in (0, 1]."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty 1D array")
    if not np.isfinite(values).all():
        raise ValueError("values contain non-finite entries")
    return rankdata(values, method="average") / len(values)


def rank_direction(treatment: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Target-free ranking direction induced by a matched model contrast."""
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    if treatment.shape != control.shape:
        raise ValueError("treatment and control must have identical shapes")
    return rank01(treatment) - rank01(control)


def orthogonalize_direction(
    direction: np.ndarray,
    against: np.ndarray | Iterable[np.ndarray],
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project a direction away from already accepted target-free directions."""
    d = np.asarray(direction, dtype=float).copy()
    if d.ndim != 1:
        raise ValueError("direction must be 1D")
    if isinstance(against, np.ndarray) and against.ndim == 1:
        matrix = against[:, None]
    elif isinstance(against, np.ndarray):
        matrix = against
    else:
        seq = [np.asarray(x, dtype=float) for x in against]
        if not seq:
            return d - d.mean() if center else d
        matrix = np.column_stack(seq)
    if matrix.shape[0] != len(d):
        raise ValueError("against directions must align row-for-row")
    if center:
        d = d - d.mean()
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    keep = np.std(matrix, axis=0) > eps
    matrix = matrix[:, keep]
    if not matrix.shape[1]:
        return d
    coef, *_ = np.linalg.lstsq(matrix, d, rcond=None)
    return d - matrix @ coef


def orthogonalize_train_test(
    direction_oof: np.ndarray,
    direction_test: np.ndarray,
    against_oof: np.ndarray | Iterable[np.ndarray],
    against_test: np.ndarray | Iterable[np.ndarray],
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a target-free OOF projection and apply the frozen transform to test."""
    d_oof = np.asarray(direction_oof, dtype=float)
    d_test = np.asarray(direction_test, dtype=float)
    if d_oof.ndim != 1 or d_test.ndim != 1:
        raise ValueError("directions must be 1D")

    def _matrix(value, n_rows):
        if isinstance(value, np.ndarray) and value.ndim == 1:
            out = value[:, None]
        elif isinstance(value, np.ndarray):
            out = value
        else:
            seq = [np.asarray(x, dtype=float) for x in value]
            out = np.column_stack(seq) if seq else np.empty((n_rows, 0), dtype=float)
        if out.shape[0] != n_rows:
            raise ValueError("against directions must align row-for-row")
        return out

    a_oof = _matrix(against_oof, len(d_oof))
    a_test = _matrix(against_test, len(d_test))
    if a_oof.shape[1] != a_test.shape[1]:
        raise ValueError("OOF/test against matrices must have the same columns")
    if not a_oof.shape[1]:
        mean = float(d_oof.mean())
        return d_oof - mean, d_test - mean

    keep = np.std(a_oof, axis=0) > eps
    a_oof = a_oof[:, keep]
    a_test = a_test[:, keep]
    if not a_oof.shape[1]:
        mean = float(d_oof.mean())
        return d_oof - mean, d_test - mean

    d_mean = float(d_oof.mean())
    a_mean = a_oof.mean(axis=0, keepdims=True)
    design = a_oof - a_mean
    coef, *_ = np.linalg.lstsq(design, d_oof - d_mean, rcond=None)
    resid_oof = (d_oof - d_mean) - design @ coef
    resid_test = (d_test - d_mean) - (a_test - a_mean) @ coef
    return resid_oof, resid_test


def apply_rank_residual(
    anchor: np.ndarray,
    direction: np.ndarray,
    weight: float,
    *,
    rerank: bool = True,
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if anchor.shape != direction.shape:
        raise ValueError("anchor and direction must have identical shapes")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and non-negative")
    score = rank01(anchor) + float(weight) * direction
    return rank01(score) if rerank else score


def _safe_auc(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, pred))


def _slice_delta(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> float:
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return _safe_auc(y[mask], candidate[mask]) - _safe_auc(y[mask], anchor[mask])


@dataclass(frozen=True)
class ResidualGate:
    """Prospective admission rule for a small ranking residual."""

    min_gain: float = 1e-6
    min_fold_wins: int = 4
    fold_tolerance: float = -2e-6
    require_id_slices: bool = True
    slice_tolerance: float = -2e-6

    def validate(self, n_folds: int) -> None:
        if not (1 <= self.min_fold_wins <= n_folds):
            raise ValueError("min_fold_wins must be between 1 and n_folds")


def evaluate_residual(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    folds: np.ndarray,
    *,
    ids: np.ndarray | None = None,
) -> dict:
    y = np.asarray(y)
    anchor = np.asarray(anchor, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    folds = np.asarray(folds)
    if not (len(y) == len(anchor) == len(candidate) == len(folds)):
        raise ValueError("y, anchor, candidate, and folds must align")

    fold_deltas = []
    for fold in np.unique(folds):
        mask = folds == fold
        fold_deltas.append(
            _safe_auc(y[mask], candidate[mask]) - _safe_auc(y[mask], anchor[mask])
        )

    anchor_auc = _safe_auc(y, anchor)
    candidate_auc = _safe_auc(y, candidate)
    result = {
        "anchor_auc": anchor_auc,
        "candidate_auc": candidate_auc,
        "gain": candidate_auc - anchor_auc,
        "fold_deltas": fold_deltas,
        "fold_wins": int(np.sum(np.asarray(fold_deltas) > 0)),
        "worst_fold_delta": float(np.min(fold_deltas)),
    }
    if ids is not None:
        ids = np.asarray(ids)
        if len(ids) != len(y):
            raise ValueError("ids must align with y")
        even = (ids.astype(np.int64) % 2) == 0
        result["id_slice_deltas"] = {
            "even": _slice_delta(y, anchor, candidate, even),
            "odd": _slice_delta(y, anchor, candidate, ~even),
        }
    return result


def _passes_gate(metrics: dict, gate: ResidualGate, n_folds: int) -> bool:
    gate.validate(n_folds)
    if metrics["gain"] < gate.min_gain:
        return False
    if metrics["fold_wins"] < gate.min_fold_wins:
        return False
    if metrics["worst_fold_delta"] < gate.fold_tolerance:
        return False
    if gate.require_id_slices:
        slices = metrics.get("id_slice_deltas")
        if slices is None:
            return False
        values = [value for value in slices.values() if np.isfinite(value)]
        if len(values) != len(slices) or min(values) < gate.slice_tolerance:
            return False
    return True


def first_passing_weight(
    y: np.ndarray,
    anchor: np.ndarray,
    direction: np.ndarray,
    folds: np.ndarray,
    weight_grid: Iterable[float],
    *,
    ids: np.ndarray | None = None,
    gate: ResidualGate | None = None,
) -> tuple[float, dict]:
    """Return the first predeclared weight satisfying the prospective gate."""
    gate = gate or ResidualGate()
    folds = np.asarray(folds)
    weights = [float(weight) for weight in weight_grid]
    if not weights or weights[0] < 0 or any(
        b <= a for a, b in zip(weights, weights[1:])
    ):
        raise ValueError("weight_grid must be strictly increasing and non-negative")
    last = None
    for weight in weights:
        candidate = apply_rank_residual(anchor, direction, weight)
        metrics = evaluate_residual(y, anchor, candidate, folds, ids=ids)
        last = metrics
        if weight == 0:
            continue
        if _passes_gate(metrics, gate, len(np.unique(folds))):
            return weight, metrics
    return 0.0, last or evaluate_residual(y, anchor, anchor, folds, ids=ids)


def rotating_residual_gate(
    y: np.ndarray,
    anchor: np.ndarray,
    direction: np.ndarray,
    folds: np.ndarray,
    weight_grid: Iterable[float],
    *,
    ids: np.ndarray | None = None,
    gate: ResidualGate | None = None,
) -> dict:
    """Choose residual weights without scoring the held outer fold.

    For each held fold, select the first passing weight on the remaining folds and
    apply that weight to the held rows.  Deployment uses the median selected weight.
    """
    y = np.asarray(y)
    anchor = np.asarray(anchor, dtype=float)
    direction = np.asarray(direction, dtype=float)
    folds = np.asarray(folds)
    ids = None if ids is None else np.asarray(ids)
    if not (len(y) == len(anchor) == len(direction) == len(folds)):
        raise ValueError("all arrays must align")
    unique_folds = np.unique(folds)
    gate = gate or ResidualGate(min_fold_wins=max(1, len(unique_folds) - 1))
    anchor_rank = rank01(anchor)
    honest_score = np.empty(len(y), dtype=float)
    selected_weights = []
    selection_metrics = []

    for held in unique_folds:
        selection = folds != held
        local_folds = folds[selection]
        local_gate = ResidualGate(
            min_gain=gate.min_gain,
            min_fold_wins=min(gate.min_fold_wins, len(np.unique(local_folds))),
            fold_tolerance=gate.fold_tolerance,
            require_id_slices=gate.require_id_slices,
            slice_tolerance=gate.slice_tolerance,
        )
        weight, metrics = first_passing_weight(
            y[selection],
            anchor[selection],
            direction[selection],
            local_folds,
            weight_grid,
            ids=None if ids is None else ids[selection],
            gate=local_gate,
        )
        selected_weights.append(weight)
        selection_metrics.append(metrics)
        held_mask = folds == held
        honest_score[held_mask] = (
            anchor_rank[held_mask] + float(weight) * direction[held_mask]
        )

    honest = rank01(honest_score)
    deploy_weight = float(np.median(selected_weights))
    honest_metrics = evaluate_residual(y, anchor, honest, folds, ids=ids)
    final_gate = ResidualGate(
        min_gain=gate.min_gain,
        min_fold_wins=min(gate.min_fold_wins, len(unique_folds)),
        fold_tolerance=gate.fold_tolerance,
        require_id_slices=gate.require_id_slices,
        slice_tolerance=gate.slice_tolerance,
    )
    accepted = (
        _passes_gate(honest_metrics, final_gate, len(unique_folds))
        and deploy_weight > 0
    )
    return {
        "accepted": bool(accepted),
        "deploy_weight": deploy_weight if accepted else 0.0,
        "selected_weight_median": deploy_weight,
        "selected_weights": selected_weights,
        "selection_metrics": selection_metrics,
        "honest_metrics": honest_metrics,
        "gate": asdict(gate),
        "honest_oof": honest,
    }
