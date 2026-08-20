from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .contrast import rank01


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score):
        raise ValueError("y and score must align")
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def validate_direction(
    oof: np.ndarray,
    test: np.ndarray,
    folds: np.ndarray,
    *,
    n_train: int,
    n_test: int,
    saved_folds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one target-free OOF/test ranking direction artifact."""
    oof = np.asarray(oof, dtype=float)
    test = np.asarray(test, dtype=float)
    folds = np.asarray(folds)
    if oof.ndim != 1 or test.ndim != 1:
        raise ValueError("direction arrays must be 1D")
    if len(oof) != n_train or len(test) != n_test:
        raise ValueError("direction lengths do not match anchor data")
    if not np.isfinite(oof).all() or not np.isfinite(test).all():
        raise ValueError("direction arrays contain non-finite values")
    if saved_folds is not None and not np.array_equal(np.asarray(saved_folds), folds):
        raise ValueError("direction artifact uses different fold assignments")
    return oof, test


def compose_rank_score(
    anchor: np.ndarray,
    directions: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    *,
    rerank: bool = False,
) -> np.ndarray:
    """Compose small target-free residual directions on top of an anchor rank."""
    base = rank01(np.asarray(anchor, dtype=float))
    score = base.copy()
    for name, direction in directions.items():
        if name not in weights:
            raise ValueError(f"missing weight for direction {name!r}")
        direction = np.asarray(direction, dtype=float)
        if direction.shape != score.shape:
            raise ValueError(f"direction {name!r} is not aligned to anchor")
        weight = float(weights[name])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weights must be finite and non-negative")
        score += weight * direction
    return rank01(score) if rerank else score


def nested_residual_selection(
    y: np.ndarray,
    anchor: np.ndarray,
    directions: Mapping[str, np.ndarray],
    folds: np.ndarray,
    weight_grids: Mapping[str, Sequence[float]],
) -> dict:
    """Select residual weights without scoring the held outer fold.

    Each held fold gets a weight vector selected only on the remaining folds.
    Ties are resolved toward smaller total residual mass. Deployment uses the
    coordinate-wise median of the held-fold selections. Zero must be available
    in every grid, so the selector can decline a direction.
    """
    y = np.asarray(y)
    folds = np.asarray(folds)
    base = rank01(np.asarray(anchor, dtype=float))
    if not (len(y) == len(base) == len(folds)):
        raise ValueError("y, anchor, and folds must align")
    names = list(directions)
    if not names:
        raise ValueError("at least one direction is required")
    dirs: dict[str, np.ndarray] = {}
    grids: list[tuple[float, ...]] = []
    for name in names:
        direction = np.asarray(directions[name], dtype=float)
        if direction.shape != base.shape or not np.isfinite(direction).all():
            raise ValueError(f"direction {name!r} is invalid")
        dirs[name] = direction
        values = tuple(float(v) for v in weight_grids[name])
        if not values or 0.0 not in values:
            raise ValueError(f"weight grid for {name!r} must include 0")
        if any(v < 0 or not np.isfinite(v) for v in values):
            raise ValueError("weight grids must be finite and non-negative")
        if len(set(values)) != len(values):
            raise ValueError("weight grids may not contain duplicates")
        grids.append(tuple(sorted(values)))

    unique_folds = np.unique(folds)
    if len(unique_folds) < 3:
        raise ValueError("at least three outer folds are required")
    honest = np.empty(len(y), dtype=float)
    selections: list[dict] = []

    for held in unique_folds:
        select_mask = folds != held
        best: tuple[float, float, tuple[float, ...]] | None = None
        for combo in itertools.product(*grids):
            score = base[select_mask].copy()
            for name, weight in zip(names, combo):
                score += weight * dirs[name][select_mask]
            score_auc = _auc(y[select_mask], score)
            residual_mass = float(sum(combo))
            key = (score_auc, -residual_mass)
            if best is None or key > (best[0], best[1]):
                best = (score_auc, -residual_mass, tuple(combo))
        assert best is not None
        selection_auc, _, combo = best
        held_mask = folds == held
        held_score = base[held_mask].copy()
        weights = {}
        for name, weight in zip(names, combo):
            held_score += weight * dirs[name][held_mask]
            weights[name] = float(weight)
        honest[held_mask] = held_score
        selections.append(
            {
                "held_fold": int(held),
                "weights": weights,
                "selection_auc": float(selection_auc),
            }
        )

    base_auc = _auc(y, base)
    honest_auc = _auc(y, honest)
    held_fold_deltas = []
    for fold in unique_folds:
        mask = folds == fold
        held_fold_deltas.append(_auc(y[mask], honest[mask]) - _auc(y[mask], base[mask]))
    accepted = bool(honest_auc > base_auc and min(held_fold_deltas) > 0)

    deploy_weights = {
        name: float(np.median([row["weights"][name] for row in selections]))
        if accepted
        else 0.0
        for name in names
    }
    deploy = compose_rank_score(base, dirs, deploy_weights)
    deploy_auc = _auc(y, deploy)
    return {
        "accepted": accepted,
        "base_auc": base_auc,
        "honest_auc": honest_auc,
        "honest_gain": honest_auc - base_auc,
        "held_fold_deltas": [float(v) for v in held_fold_deltas],
        "selections": selections,
        "deploy_weights": deploy_weights,
        "deploy_oof_auc": deploy_auc,
        "deploy_oof_gain": deploy_auc - base_auc,
        "honest_oof": honest,
        "deploy_oof": deploy,
    }


def stability_diagnostics(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    ids: np.ndarray,
    *,
    moduli: Sequence[int] = (2, 3, 5, 7, 11),
    contiguous_blocks: Sequence[int] = (5, 10, 20),
) -> dict:
    """Stress-test AUC gain across modulo and contiguous sorted-ID slices."""
    y = np.asarray(y)
    anchor = np.asarray(anchor, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    ids = np.asarray(ids, dtype=np.int64)
    if not (len(y) == len(anchor) == len(candidate) == len(ids)):
        raise ValueError("all arrays must align")

    out: dict[str, dict] = {"modulo": {}, "contiguous": {}}
    for modulus in moduli:
        values = []
        for residue in range(int(modulus)):
            mask = (ids % modulus) == residue
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            values.append(_auc(y[mask], candidate[mask]) - _auc(y[mask], anchor[mask]))
        if values:
            out["modulo"][str(modulus)] = {
                "min": float(min(values)),
                "max": float(max(values)),
                "wins": int(sum(v > 0 for v in values)),
                "total": len(values),
            }

    order = np.argsort(ids, kind="stable")
    for n_blocks in contiguous_blocks:
        values = []
        for idx in np.array_split(order, int(n_blocks)):
            if len(idx) < 20 or len(np.unique(y[idx])) < 2:
                continue
            values.append(_auc(y[idx], candidate[idx]) - _auc(y[idx], anchor[idx]))
        if values:
            out["contiguous"][str(n_blocks)] = {
                "min": float(min(values)),
                "max": float(max(values)),
                "wins": int(sum(v > 0 for v in values)),
                "total": len(values),
            }
    return out


def all_stability_slices_positive(diagnostics: Mapping[str, Mapping[str, Mapping[str, float]]]) -> bool:
    """Return True only when every reported stress slice has positive gain."""
    seen = False
    for family in diagnostics.values():
        for metrics in family.values():
            seen = True
            if int(metrics["wins"]) != int(metrics["total"]) or float(metrics["min"]) <= 0:
                return False
    return seen
