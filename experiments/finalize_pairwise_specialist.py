#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from s6e8.config import ID_COL, TARGET
from s6e8.contrast import ResidualGate, evaluate_residual, rank01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply(
    anchor: np.ndarray,
    direction: np.ndarray,
    coefficient: float,
    mask: np.ndarray,
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=float)
    direction = np.asarray(direction, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not (anchor.shape == direction.shape == mask.shape):
        raise ValueError("anchor, direction, and mask must align")
    return rank01(rank01(anchor) + float(coefficient) * direction * mask)


def _passes(metrics: dict, gate: ResidualGate, n_folds: int) -> bool:
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
        values = [float(value) for value in slices.values() if np.isfinite(value)]
        if len(values) != len(slices) or min(values) < gate.slice_tolerance:
            return False
    return True


def _candidate_grid(
    cutoffs: list[float],
    magnitudes: list[float],
):
    # Order is fixed prospectively: the narrowest specialist first, then the
    # smallest coefficient, with the natural + orientation evaluated before -.
    for cutoff in cutoffs:
        for magnitude in magnitudes:
            yield cutoff, +float(magnitude)
            yield cutoff, -float(magnitude)


def _rotating_specialist_gate(
    y: np.ndarray,
    ids: np.ndarray,
    folds: np.ndarray,
    anchor: np.ndarray,
    direction: np.ndarray,
    *,
    cutoffs: list[float],
    magnitudes: list[float],
    gate: ResidualGate,
) -> dict:
    unique_folds = np.unique(folds)
    anchor_rank = rank01(anchor)
    honest_score = np.empty(len(y), dtype=float)
    selected = []
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
        chosen_cutoff = None
        chosen_coefficient = 0.0
        chosen_metrics = evaluate_residual(
            y[selection],
            anchor[selection],
            anchor[selection],
            local_folds,
            ids=ids[selection],
        )
        for cutoff, coefficient in _candidate_grid(cutoffs, magnitudes):
            local_mask = anchor_rank[selection] <= cutoff
            candidate = _apply(
                anchor[selection],
                direction[selection],
                coefficient,
                local_mask,
            )
            metrics = evaluate_residual(
                y[selection],
                anchor[selection],
                candidate,
                local_folds,
                ids=ids[selection],
            )
            if _passes(metrics, local_gate, len(np.unique(local_folds))):
                chosen_cutoff = float(cutoff)
                chosen_coefficient = float(coefficient)
                chosen_metrics = metrics
                break

        selected.append(
            {
                "held_fold": int(held),
                "cutoff": chosen_cutoff,
                "coefficient": chosen_coefficient,
            }
        )
        selection_metrics.append(chosen_metrics)
        held_mask = folds == held
        if chosen_cutoff is None or chosen_coefficient == 0.0:
            honest_score[held_mask] = anchor_rank[held_mask]
        else:
            specialist = anchor_rank[held_mask] <= chosen_cutoff
            honest_score[held_mask] = (
                anchor_rank[held_mask]
                + chosen_coefficient * direction[held_mask] * specialist
            )

    honest = rank01(honest_score)
    honest_metrics = evaluate_residual(y, anchor, honest, folds, ids=ids)
    nonzero = [row for row in selected if row["coefficient"] != 0.0]
    if not nonzero:
        deploy_cutoff = None
        deploy_coefficient = 0.0
    else:
        deploy_cutoff = float(np.median([row["cutoff"] for row in nonzero]))
        deploy_coefficient = float(
            np.median([row["coefficient"] for row in nonzero])
        )

    if deploy_cutoff is None or deploy_coefficient == 0.0:
        deploy_candidate = anchor.copy()
    else:
        deploy_candidate = _apply(
            anchor,
            direction,
            deploy_coefficient,
            anchor_rank <= deploy_cutoff,
        )
    deploy_metrics = evaluate_residual(
        y, anchor, deploy_candidate, folds, ids=ids
    )
    final_gate = ResidualGate(
        min_gain=gate.min_gain,
        min_fold_wins=min(gate.min_fold_wins, len(unique_folds)),
        fold_tolerance=gate.fold_tolerance,
        require_id_slices=gate.require_id_slices,
        slice_tolerance=gate.slice_tolerance,
    )
    accepted = bool(
        nonzero
        and _passes(honest_metrics, final_gate, len(unique_folds))
        and _passes(deploy_metrics, final_gate, len(unique_folds))
    )
    return {
        "accepted": accepted,
        "selected": selected,
        "selection_metrics": selection_metrics,
        "deploy_cutoff": deploy_cutoff if accepted else None,
        "deploy_coefficient": deploy_coefficient if accepted else 0.0,
        "honest_metrics": honest_metrics,
        "deploy_metrics": deploy_metrics,
        "gate": asdict(gate),
        "honest_oof": honest,
        "deploy_oof": deploy_candidate,
    }


def _slice_metrics(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    correction: np.ndarray,
    direction: np.ndarray,
) -> dict:
    anchor_rank = rank01(anchor)
    correction_strength = rank01(np.abs(correction - np.median(correction)))
    direction_strength = rank01(np.abs(direction))
    masks = {
        "anchor_low": anchor_rank <= 0.20,
        "anchor_boundary": (anchor_rank > 0.35) & (anchor_rank <= 0.65),
        "anchor_high": anchor_rank > 0.80,
        "correction_high": correction_strength > 0.75,
        "direction_high": direction_strength > 0.75,
    }
    rows = {}
    for name, mask in masks.items():
        if len(np.unique(y[mask])) < 2:
            rows[name] = None
            continue
        local = evaluate_residual(
            y[mask],
            anchor[mask],
            candidate[mask],
            np.zeros(int(mask.sum()), dtype=np.int8),
        )
        rows[name] = float(local["gain"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize the cross-fitted pairwise direction as a low-score specialist."
    )
    parser.add_argument("--pairwise-dir", required=True)
    parser.add_argument("--cutoffs", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument(
        "--magnitudes",
        default="0.00025,0.0005,0.001,0.0025,0.005,0.0075,0.01",
    )
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--min-fold-wins", type=int, default=4)
    parser.add_argument("--fold-tolerance", type=float, default=-2e-6)
    parser.add_argument("--slice-tolerance", type=float, default=-2e-6)
    parser.add_argument("--structural-tolerance", type=float, default=-5e-5)
    parser.add_argument("--out-dir", default="artifacts/pairwise_specialist")
    args = parser.parse_args()

    pairwise = Path(args.pairwise_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(pairwise / "oof.csv")
    required = {
        ID_COL,
        TARGET,
        "fold",
        "champion",
        "pairwise_correction",
        "direction",
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"pairwise OOF is missing columns: {missing}")

    y = oof[TARGET].to_numpy(np.int8)
    ids = oof[ID_COL].to_numpy()
    folds = oof["fold"].to_numpy(int)
    anchor = oof["champion"].to_numpy(float)
    correction = oof["pairwise_correction"].to_numpy(float)
    direction = oof["direction"].to_numpy(float)
    cutoffs = [float(value) for value in args.cutoffs.split(",") if value.strip()]
    magnitudes = [
        float(value) for value in args.magnitudes.split(",") if value.strip()
    ]
    if not cutoffs or any(not 0.0 < value < 1.0 for value in cutoffs):
        raise ValueError("cutoffs must lie strictly inside (0, 1)")
    if any(b <= a for a, b in zip(cutoffs, cutoffs[1:])):
        raise ValueError("cutoffs must be strictly increasing")
    if not magnitudes or any(value <= 0 for value in magnitudes):
        raise ValueError("magnitudes must be positive")
    if any(b <= a for a, b in zip(magnitudes, magnitudes[1:])):
        raise ValueError("magnitudes must be strictly increasing")

    gate = ResidualGate(
        min_gain=args.min_gain,
        min_fold_wins=args.min_fold_wins,
        fold_tolerance=args.fold_tolerance,
        require_id_slices=True,
        slice_tolerance=args.slice_tolerance,
    )
    decision = _rotating_specialist_gate(
        y,
        ids,
        folds,
        anchor,
        direction,
        cutoffs=cutoffs,
        magnitudes=magnitudes,
        gate=gate,
    )
    honest = decision.pop("honest_oof")
    deploy_oof = decision.pop("deploy_oof")
    structural = _slice_metrics(y, anchor, deploy_oof, correction, direction)
    finite_structural = [value for value in structural.values() if value is not None]
    structural_pass = bool(
        finite_structural and min(finite_structural) >= args.structural_tolerance
    )
    accepted = bool(decision["accepted"] and structural_pass)

    champion_test = pd.read_csv(pairwise / "submission_promoted.csv")
    direction_test = np.load(pairwise / "direction_test.npy")
    if (
        len(champion_test) != len(direction_test)
        or ID_COL not in champion_test
        or TARGET not in champion_test
    ):
        raise ValueError("pairwise test artifact does not align")
    test_anchor = pd.to_numeric(champion_test[TARGET], errors="coerce").to_numpy(float)
    if not np.isfinite(test_anchor).all() or not np.isfinite(direction_test).all():
        raise ValueError("test artifact contains non-finite values")

    if accepted:
        cutoff = float(decision["deploy_cutoff"])
        coefficient = float(decision["deploy_coefficient"])
        test_mask = rank01(test_anchor) <= cutoff
        test_prediction = _apply(
            test_anchor, direction_test, coefficient, test_mask
        )
    else:
        cutoff = None
        coefficient = 0.0
        test_prediction = test_anchor.copy()

    submission = champion_test[[ID_COL]].copy()
    submission[TARGET] = test_prediction
    submission_path = out / "submission_pairwise_specialist.csv"
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            ID_COL: ids,
            TARGET: y,
            "fold": folds,
            "champion": anchor,
            "honest_candidate": honest,
            "deploy_candidate": deploy_oof,
            "direction": direction,
        }
    ).to_csv(out / "oof.csv", index=False)

    report = {
        "version": "pairwise-low-tail-specialist-v1",
        "hypothesis": (
            "the pairwise correction is useful as a low-score specialist rather "
            "than a global endpoint; cutoff, sign, and magnitude are selected only "
            "on non-held folds"
        ),
        "candidate_grid": {
            "cutoffs": cutoffs,
            "magnitudes": magnitudes,
            "sign_order": [1, -1],
            "ordering": "cutoff_then_magnitude_then_sign_first_passing",
        },
        "decision": decision,
        "structural_deltas": structural,
        "structural_tolerance": args.structural_tolerance,
        "structural_pass": structural_pass,
        "accepted": accepted,
        "deploy_cutoff": cutoff,
        "deploy_coefficient": coefficient,
        "submission": {
            "path": str(submission_path),
            "rows": int(len(submission)),
            "unique_predictions": int(submission[TARGET].nunique()),
            "sha256": _sha256(submission_path),
        },
    }
    (out / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
