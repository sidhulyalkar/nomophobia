#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.config import ID_COL, TARGET
from s6e8.contrast import (
    ResidualGate,
    apply_rank_residual,
    orthogonalize_train_test,
    rotating_residual_gate,
)
from s6e8.submission import write_submission
from s6e8.validation import validate_competition_frames

DEFAULT_WEIGHT_GRID = [
    0.0, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025,
    0.0035, 0.005, 0.0075, 0.010, 0.015, 0.020,
]


def _parse_weights(text: str | None) -> list[float]:
    if text is None:
        return DEFAULT_WEIGHT_GRID.copy()
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("weight grid is empty")
    if values[0] != 0.0:
        values = [0.0, *values]
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("weights must be strictly increasing")
    return values


def _load_competition(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    validate_competition_frames(train, test, sample)
    return train, test, sample


def _load_anchor(train, test, oof_path, test_path, oof_col, test_col, fold_col):
    oof = pd.read_csv(oof_path)
    pred_test = pd.read_csv(test_path)
    if not np.array_equal(oof[ID_COL].to_numpy(), train[ID_COL].to_numpy()):
        raise ValueError("anchor OOF IDs do not align with train")
    if not np.array_equal(pred_test[ID_COL].to_numpy(), test[ID_COL].to_numpy()):
        raise ValueError("anchor test IDs do not align with test")
    y = train[TARGET].to_numpy(np.int8)
    if TARGET in oof and not np.array_equal(oof[TARGET].to_numpy(np.int8), y):
        raise ValueError("anchor OOF target does not match train")
    return (
        oof[oof_col].to_numpy(float),
        pred_test[test_col].to_numpy(float),
        oof[fold_col].to_numpy(int),
    )


def _load_direction(directory: Path, n_train: int, n_test: int):
    oof = np.load(directory / "direction_oof.npy")
    test = np.load(directory / "direction_test.npy")
    if len(oof) != n_train or len(test) != n_test:
        raise ValueError(f"direction length mismatch in {directory}")
    meta = {}
    decision = directory / "decision.json"
    if decision.exists():
        try:
            meta = json.loads(decision.read_text())
        except json.JSONDecodeError:
            meta = {}
    return np.asarray(oof, float), np.asarray(test, float), meta


def _standardize(oof: np.ndarray, test: np.ndarray):
    scale = float(np.std(oof))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("direction has near-zero OOF variance")
    mean = float(np.mean(oof))
    return (oof - mean) / scale, (test - mean) / scale, mean, scale


def _pairwise_corr(matrix: np.ndarray) -> list[list[float]]:
    if matrix.shape[1] == 1:
        return [[1.0]]
    return np.corrcoef(matrix, rowvar=False).tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Compose accepted target-free residual directions into one prospective axis."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--anchor-oof", required=True)
    parser.add_argument("--anchor-test", required=True)
    parser.add_argument("--anchor-oof-col", default="honest_blend")
    parser.add_argument("--anchor-test-col", default=TARGET)
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--direction-dirs", nargs="+", required=True)
    parser.add_argument(
        "--mode",
        choices=["equal_standardized", "sequential_orthogonal"],
        default="sequential_orthogonal",
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--min-fold-wins", type=int, default=4)
    parser.add_argument("--fold-tolerance", type=float, default=-2e-6)
    parser.add_argument("--slice-tolerance", type=float, default=-2e-6)
    parser.add_argument("--out-dir", default="artifacts/frontier_composite")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, test, sample = _load_competition(data_dir)
    y = train[TARGET].to_numpy(np.int8)
    ids = train[ID_COL].to_numpy()
    anchor_oof, anchor_test, folds = _load_anchor(
        train,
        test,
        Path(args.anchor_oof),
        Path(args.anchor_test),
        args.anchor_oof_col,
        args.anchor_test_col,
        args.fold_col,
    )

    raw_oof = []
    raw_test = []
    source_meta = []
    names = []
    for text in args.direction_dirs:
        directory = Path(text)
        oof_direction, test_direction, meta = _load_direction(
            directory, len(train), len(test)
        )
        raw_oof.append(oof_direction)
        raw_test.append(test_direction)
        names.append(directory.name)
        source_meta.append(
            {
                "name": directory.name,
                "path": str(directory),
                "family": meta.get("family"),
                "treatment": meta.get("treatment"),
                "residual_gate": meta.get("residual_gate"),
            }
        )

    raw_matrix = np.column_stack(raw_oof)
    processed_oof = []
    processed_test = []
    transforms = []

    for name, oof_direction, test_direction in zip(names, raw_oof, raw_test):
        if args.mode == "sequential_orthogonal" and processed_oof:
            residual_oof, residual_test = orthogonalize_train_test(
                oof_direction,
                test_direction,
                np.column_stack(processed_oof),
                np.column_stack(processed_test),
            )
            transform_type = "orthogonalize_then_standardize"
        else:
            residual_oof, residual_test = oof_direction, test_direction
            transform_type = "standardize"
        z_oof, z_test, mean, scale = _standardize(residual_oof, residual_test)
        processed_oof.append(z_oof)
        processed_test.append(z_test)
        transforms.append(
            {"name": name, "type": transform_type, "mean": mean, "scale": scale}
        )

    composite_oof = np.mean(np.column_stack(processed_oof), axis=1)
    composite_test = np.mean(np.column_stack(processed_test), axis=1)
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
        composite_oof,
        folds,
        _parse_weights(args.weights),
        ids=ids,
        gate=gate,
    )
    honest = decision.pop("honest_oof")
    raw_weight = float(decision["selected_weight_median"])
    candidate_test = apply_rank_residual(anchor_test, composite_test, raw_weight)
    gated_test = (
        apply_rank_residual(
            anchor_test, composite_test, float(decision["deploy_weight"])
        )
        if decision["accepted"]
        else anchor_test.copy()
    )

    np.save(out / "direction_oof.npy", composite_oof)
    np.save(out / "direction_test.npy", composite_test)
    np.save(out / "oof_honest_residual.npy", honest)
    candidate_stats = write_submission(
        out / "submission_candidate.csv", sample, test, candidate_test
    )
    gated_stats = write_submission(
        out / "submission_gated.csv", sample, test, gated_test
    )

    report = {
        "version": "frontier-direction-composer-v1",
        "mode": args.mode,
        "sources": source_meta,
        "raw_direction_correlation": _pairwise_corr(raw_matrix),
        "processed_direction_correlation": _pairwise_corr(
            np.column_stack(processed_oof)
        ),
        "transforms": transforms,
        "anchor_auc": float(roc_auc_score(y, anchor_oof)),
        "residual_gate": decision,
        "candidate_submission": candidate_stats,
        "gated_submission": gated_stats,
    }
    (out / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
