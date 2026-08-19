#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.config import ID_COL, RAW_COLS, TARGET
from s6e8.contrast import ResidualGate, apply_rank_residual, rank_direction, rotating_residual_gate
from s6e8.generator_graph import cross_fitted_graph_scores
from s6e8.submission import write_submission
from s6e8.validation import validate_competition_frames

DEFAULT_WEIGHTS = [
    0.0,
    0.0025,
    0.005,
    0.0075,
    0.010,
    0.015,
    0.025,
    0.040,
    0.060,
    0.080,
    0.120,
    0.160,
    0.200,
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_weights(text: str | None) -> list[float]:
    if text is None:
        return DEFAULT_WEIGHTS.copy()
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("weight grid is empty")
    if values[0] != 0.0:
        values = [0.0, *values]
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("weights must be strictly increasing")
    return values


def _load_anchor(
    oof_path: Path,
    test_path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    oof_col: str,
    test_col: str,
    fold_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oof = pd.read_csv(oof_path)
    pred_test = pd.read_csv(test_path)
    required_oof = {ID_COL, fold_col, oof_col}
    required_test = {ID_COL, test_col}
    if not required_oof.issubset(oof.columns):
        raise ValueError(f"anchor OOF missing {sorted(required_oof - set(oof.columns))}")
    if not required_test.issubset(pred_test.columns):
        raise ValueError(f"anchor test missing {sorted(required_test - set(pred_test.columns))}")
    if not np.array_equal(oof[ID_COL].to_numpy(), train[ID_COL].to_numpy()):
        raise ValueError("anchor OOF IDs are not aligned to train")
    if not np.array_equal(pred_test[ID_COL].to_numpy(), test[ID_COL].to_numpy()):
        raise ValueError("anchor test IDs are not aligned to test")
    return (
        pd.to_numeric(oof[oof_col]).to_numpy(float),
        pd.to_numeric(pred_test[test_col]).to_numpy(float),
        pd.to_numeric(oof[fold_col]).to_numpy(int),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe latent generator graph contrast: exact-value evidence control "
            "versus higher-order source-neighborhood token evidence."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--anchor-oof", required=True)
    parser.add_argument("--anchor-test", required=True)
    parser.add_argument("--anchor-oof-col", default="honest_blend")
    parser.add_argument("--anchor-test-col", default=TARGET)
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--smoothing", type=float, default=10.0)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--min-fold-wins", type=int, default=4)
    parser.add_argument("--fold-tolerance", type=float, default=-2e-6)
    parser.add_argument("--slice-tolerance", type=float, default=-2e-6)
    parser.add_argument("--out-dir", default="artifacts/generator_graph_frontier")
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    validate_competition_frames(train, test, sample)
    missing = [column for column in RAW_COLS if column not in train or column not in test]
    if missing:
        raise ValueError(f"missing raw predictors: {missing}")

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

    graph = cross_fitted_graph_scores(
        train[RAW_COLS],
        test[RAW_COLS],
        y,
        folds,
        smoothing=args.smoothing,
        min_support=args.min_support,
    )
    oof_control = np.asarray(graph["oof_control"], float)
    oof_treatment = np.asarray(graph["oof_treatment"], float)
    test_control = np.asarray(graph["test_control"], float)
    test_treatment = np.asarray(graph["test_treatment"], float)

    control_auc = float(roc_auc_score(y, oof_control))
    treatment_auc = float(roc_auc_score(y, oof_treatment))
    direction_oof = rank_direction(oof_treatment, oof_control)
    direction_test = rank_direction(test_treatment, test_control)
    weights = _parse_weights(args.weights)
    gate = ResidualGate(
        min_gain=args.min_gain,
        min_fold_wins=args.min_fold_wins,
        fold_tolerance=args.fold_tolerance,
        require_id_slices=True,
        slice_tolerance=args.slice_tolerance,
    )
    decision = rotating_residual_gate(
        y,
        anchor_oof,
        direction_oof,
        folds,
        weights,
        ids=ids,
        gate=gate,
    )
    honest = np.asarray(decision.pop("honest_oof"), float)
    deploy_weight = float(decision["deploy_weight"])
    candidate_test = (
        apply_rank_residual(anchor_test, direction_test, deploy_weight)
        if decision["accepted"]
        else anchor_test.copy()
    )
    candidate_oof = (
        apply_rank_residual(anchor_oof, direction_oof, deploy_weight)
        if decision["accepted"]
        else anchor_oof.copy()
    )

    np.save(out / "oof_control.npy", oof_control)
    np.save(out / "oof_treatment.npy", oof_treatment)
    np.save(out / "test_control.npy", test_control)
    np.save(out / "test_treatment.npy", test_treatment)
    np.save(out / "direction_oof.npy", direction_oof)
    np.save(out / "direction_test.npy", direction_test)
    np.save(out / "honest_candidate_oof.npy", honest)

    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "fold": folds,
            "anchor": anchor_oof,
            "graph_control": oof_control,
            "graph_treatment": oof_treatment,
            "graph_direction": direction_oof,
            "candidate_frozen": candidate_oof,
            "candidate_honest": honest,
        }
    ).to_csv(out / "oof_generator_graph.csv", index=False)

    submission_path = out / "submission_generator_graph.csv"
    write_submission(test[ID_COL], candidate_test, submission_path)

    result = {
        "version": "generator-graph-v1",
        "hypothesis": (
            "The synthetic generator preserves higher-order source-row identity through "
            "joint value collisions; cross-fitted joint-token label evidence supplies an "
            "anchor-orthogonal ranking correction beyond univariate exact-value TE."
        ),
        "rows": {"train": len(train), "test": len(test)},
        "smoothing": args.smoothing,
        "min_support": args.min_support,
        "control_auc": control_auc,
        "treatment_auc": treatment_auc,
        "treatment_minus_control_auc": treatment_auc - control_auc,
        "direction_rank_corr_with_anchor": float(
            pd.Series(direction_oof).corr(pd.Series(anchor_oof), method="spearman")
        ),
        "control_specs": graph["control_specs"],
        "treatment_specs": graph["treatment_specs"],
        "fold_diagnostics": graph["fold_diagnostics"],
        "residual_decision": decision,
        "frozen_candidate_auc": float(roc_auc_score(y, candidate_oof)),
        "honest_candidate_auc": float(roc_auc_score(y, honest)),
        "anchor_auc": float(roc_auc_score(y, anchor_oof)),
        "elapsed_seconds": round(time.time() - started, 2),
        "submission": {
            "path": str(submission_path),
            "sha256": _sha256(submission_path),
            "accepted": bool(decision["accepted"]),
            "deploy_weight": deploy_weight,
        },
    }
    (out / "decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
