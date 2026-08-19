#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frontier_contrast_campaign import (
    _load_anchor,
    _parse_weights,
    _read_competition,
)
from s6e8.config import ID_COL, RAW_COLS, TARGET
from s6e8.contrast import (
    ResidualGate,
    apply_rank_residual,
    rank01,
    rank_direction,
    rotating_residual_gate,
)
from s6e8.generative_evidence import fit_tan_model, score_tan_model, tan_report
from s6e8.submission import write_submission


def _safe_delta(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    if int(mask.sum()) < 200 or len(np.unique(y[mask])) < 2:
        return None
    return float(
        roc_auc_score(y[mask], candidate[mask])
        - roc_auc_score(y[mask], anchor[mask])
    )


def _structural_stability(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    train: pd.DataFrame,
    tan_score: np.ndarray,
    naive_score: np.ndarray,
) -> dict:
    anchor_rank = rank01(anchor)
    disagreement = np.abs(rank01(tan_score) - anchor_rank)
    disagreement_rank = rank01(disagreement)
    dependence = np.abs(rank01(tan_score) - rank01(naive_score))
    dependence_rank = rank01(dependence)
    missing = train[RAW_COLS].isna().sum(axis=1).to_numpy(int)

    masks = {
        "anchor_low": anchor_rank <= 0.20,
        "anchor_boundary": (anchor_rank > 0.35) & (anchor_rank <= 0.65),
        "anchor_high": anchor_rank > 0.80,
        "tan_disagreement_low": disagreement_rank <= 0.50,
        "tan_disagreement_high": disagreement_rank > 0.75,
        "dependency_low": dependence_rank <= 0.50,
        "dependency_high": dependence_rank > 0.75,
        "missing_0": missing == 0,
        "missing_1": missing == 1,
        "missing_2plus": missing >= 2,
    }
    deltas = {
        name: _safe_delta(y, anchor, candidate, mask)
        for name, mask in masks.items()
    }
    finite = [value for value in deltas.values() if value is not None]
    return {
        "deltas": deltas,
        "worst_delta": None if not finite else float(min(finite)),
        "wins": int(sum(value > 0 for value in finite)),
        "total": len(finite),
    }


def _gate_direction(
    *,
    name: str,
    y: np.ndarray,
    ids: np.ndarray,
    folds: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    direction_oof: np.ndarray,
    direction_test: np.ndarray,
    weights: list[float],
    gate: ResidualGate,
    train: pd.DataFrame,
    tan_oof: np.ndarray,
    naive_oof: np.ndarray,
    structural_tolerance: float,
    out: Path,
    sample: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    decision = rotating_residual_gate(
        y,
        anchor_oof,
        direction_oof,
        folds,
        weights,
        ids=ids,
        gate=gate,
    )
    honest = decision.pop("honest_oof")
    structural = _structural_stability(
        y, anchor_oof, honest, train, tan_oof, naive_oof
    )
    structural_pass = bool(
        structural["worst_delta"] is not None
        and structural["worst_delta"] >= structural_tolerance
    )
    accepted = bool(decision["accepted"] and structural_pass)
    deploy_weight = float(decision["deploy_weight"]) if accepted else 0.0
    diagnostic_weight = float(decision["selected_weight_median"])
    diagnostic_test = apply_rank_residual(
        anchor_test, direction_test, diagnostic_weight
    )
    gated_test = (
        apply_rank_residual(anchor_test, direction_test, deploy_weight)
        if accepted
        else anchor_test.copy()
    )
    diagnostic_stats = write_submission(
        out / f"submission_{name}_diagnostic.csv",
        sample,
        test,
        diagnostic_test,
    )
    gated_stats = write_submission(
        out / f"submission_{name}_gated.csv",
        sample,
        test,
        gated_test,
    )
    return {
        "name": name,
        "accepted": accepted,
        "deploy_weight": deploy_weight,
        "residual_gate": decision,
        "structural_stability": structural,
        "structural_tolerance": structural_tolerance,
        "diagnostic_submission": diagnostic_stats,
        "gated_submission": gated_stats,
        "honest_oof": honest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit a Tree-Augmented Naive Bayes model and test whether its "
            "class-conditional dependency evidence adds a new ranking basis."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--anchor-oof", required=True)
    parser.add_argument("--anchor-test", required=True)
    parser.add_argument("--anchor-oof-col", default="honest_blend")
    parser.add_argument("--anchor-test-col", default=TARGET)
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--n-bins", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--min-fold-wins", type=int, default=4)
    parser.add_argument("--fold-tolerance", type=float, default=-2e-6)
    parser.add_argument("--slice-tolerance", type=float, default=-2e-6)
    parser.add_argument("--structural-tolerance", type=float, default=-5e-5)
    parser.add_argument("--out-dir", default="artifacts/generative_evidence")
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = _parse_weights(args.weights)

    train, test, sample = _read_competition(data_dir)
    y = train[TARGET].to_numpy(np.int8)
    ids = train[ID_COL].to_numpy()
    anchor_oof, anchor_test, folds = _load_anchor(
        Path(args.anchor_oof),
        Path(args.anchor_test),
        train,
        test,
        oof_col=args.anchor_oof_col,
        test_col=args.anchor_test_col,
        fold_col=args.fold_col,
    )
    unique_folds = np.unique(folds)
    if len(unique_folds) < 3:
        raise ValueError("at least three aligned outer folds are required")

    components = ("naive", "tan", "dependency")
    oof = {name: np.empty(len(train), dtype=float) for name in components}
    test_pred = {name: np.zeros(len(test), dtype=float) for name in components}
    fold_reports = []

    for fold in unique_folds:
        fold_started = time.time()
        train_idx = np.where(folds != fold)[0]
        valid_idx = np.where(folds == fold)[0]
        model = fit_tan_model(
            train.iloc[train_idx],
            y[train_idx],
            columns=RAW_COLS,
            n_bins=args.n_bins,
            alpha=args.alpha,
        )
        valid_scores = score_tan_model(model, train.iloc[valid_idx])
        test_scores = score_tan_model(model, test)
        for name in components:
            # Rank-normalize fold outputs so independently fitted generative
            # models have a common scale before they are stitched into OOF.
            oof[name][valid_idx] = rank01(valid_scores[name])
            test_pred[name] += rank01(test_scores[name]) / len(unique_folds)

        row = {
            "fold": int(fold),
            "naive_auc": float(roc_auc_score(y[valid_idx], valid_scores["naive"])),
            "tan_auc": float(roc_auc_score(y[valid_idx], valid_scores["tan"])),
            "dependency_auc": float(
                roc_auc_score(y[valid_idx], valid_scores["dependency"])
            ),
            "tan_minus_naive": float(
                roc_auc_score(y[valid_idx], valid_scores["tan"])
                - roc_auc_score(y[valid_idx], valid_scores["naive"])
            ),
            "model": tan_report(model),
            "seconds": round(time.time() - fold_started, 2),
        }
        fold_reports.append(row)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"model"}
                }
            ),
            flush=True,
        )

    for name in components:
        np.save(out / f"oof_{name}.npy", oof[name])
        np.save(out / f"test_{name}.npy", test_pred[name])

    directions = {
        # Primary hypothesis: class-conditional feature dependencies add evidence
        # beyond the marginal generative model.
        "dependency_contrast": (
            rank_direction(oof["tan"], oof["naive"]),
            rank_direction(test_pred["tan"], test_pred["naive"]),
        ),
        # Secondary falsification: the complete TAN ranking itself may be useful
        # even if the matched dependency contrast is too small.
        "tan_direct": (
            rank_direction(oof["tan"], anchor_oof),
            rank_direction(test_pred["tan"], anchor_test),
        ),
        "naive_direct": (
            rank_direction(oof["naive"], anchor_oof),
            rank_direction(test_pred["naive"], anchor_test),
        ),
    }

    gate = ResidualGate(
        min_gain=args.min_gain,
        min_fold_wins=args.min_fold_wins,
        fold_tolerance=args.fold_tolerance,
        require_id_slices=True,
        slice_tolerance=args.slice_tolerance,
    )
    decisions = {}
    honest_columns = {}
    for name, (direction_oof, direction_test) in directions.items():
        np.save(out / f"direction_{name}_oof.npy", direction_oof)
        np.save(out / f"direction_{name}_test.npy", direction_test)
        decision = _gate_direction(
            name=name,
            y=y,
            ids=ids,
            folds=folds,
            anchor_oof=anchor_oof,
            anchor_test=anchor_test,
            direction_oof=direction_oof,
            direction_test=direction_test,
            weights=weights,
            gate=gate,
            train=train,
            tan_oof=oof["tan"],
            naive_oof=oof["naive"],
            structural_tolerance=args.structural_tolerance,
            out=out,
            sample=sample,
            test=test,
        )
        honest_columns[name] = decision.pop("honest_oof")
        decisions[name] = decision

    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "fold": folds,
            "anchor": anchor_oof,
            "naive": oof["naive"],
            "tan": oof["tan"],
            "dependency": oof["dependency"],
            **{f"honest_{name}": values for name, values in honest_columns.items()},
        }
    ).to_csv(out / "oof.csv", index=False)

    # Promotion priority is prospective, not chosen after seeing pooled OOF:
    # dependency contrast first, then full TAN, then marginal NB.
    priority = ["dependency_contrast", "tan_direct", "naive_direct"]
    promoted = next(
        (name for name in priority if decisions[name]["accepted"]), None
    )
    if promoted is None:
        promoted_test = anchor_test.copy()
    else:
        direction_test = directions[promoted][1]
        promoted_test = apply_rank_residual(
            anchor_test,
            direction_test,
            float(decisions[promoted]["deploy_weight"]),
        )
    promoted_stats = write_submission(
        out / "submission_promoted.csv", sample, test, promoted_test
    )

    report = {
        "version": "generative-evidence-tan-v1",
        "hypothesis": (
            "a class-conditional dependency likelihood ratio supplies ranking "
            "information that discriminative tree ensembles do not represent"
        ),
        "n_bins": args.n_bins,
        "alpha": args.alpha,
        "weights": weights,
        "anchor_auc": float(roc_auc_score(y, anchor_oof)),
        "standalone": {
            "naive_auc": float(roc_auc_score(y, oof["naive"])),
            "tan_auc": float(roc_auc_score(y, oof["tan"])),
            "dependency_auc": float(roc_auc_score(y, oof["dependency"])),
            "tan_minus_naive": float(
                roc_auc_score(y, oof["tan"])
                - roc_auc_score(y, oof["naive"])
            ),
            "anchor_tan_rank_corr": float(
                pd.Series(anchor_oof).rank(method="average").corr(
                    pd.Series(oof["tan"]).rank(method="average")
                )
            ),
        },
        "fold_reports": fold_reports,
        "directions": decisions,
        "promotion_priority": priority,
        "promoted": promoted,
        "promoted_submission": promoted_stats,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
