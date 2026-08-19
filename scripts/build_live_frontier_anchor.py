#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from s6e8.config import ID_COL, TARGET
from s6e8.submission import write_submission

STREAMS = ["lgb_s10_i5", "xgb_s10_i5", "lgb_s10_i10", "lgb_s20_i5"]
DEFAULT_MEAN_WEIGHTS = np.array([0.15, 0.44, 0.26, 0.15], dtype=float)
DEFAULT_ROTATION_WEIGHTS = {
    0: np.array([0.125, 0.45, 0.30, 0.125], dtype=float),
    1: np.array([0.15, 0.45, 0.25, 0.15], dtype=float),
    2: np.array([0.15, 0.45, 0.25, 0.15], dtype=float),
    3: np.array([0.175, 0.40, 0.25, 0.175], dtype=float),
    4: np.array([0.15, 0.45, 0.25, 0.15], dtype=float),
}


def _rank01(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prediction stream contains non-finite values")
    return rankdata(values, method="average") / len(values)


def _read_oof(path: str, prediction_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [ID_COL, TARGET, "fold", prediction_column]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return frame[required].rename(columns={prediction_column: "prediction"})


def _read_test(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if ID_COL not in frame or TARGET not in frame:
        raise ValueError(f"{path} must contain {ID_COL!r} and {TARGET!r}")
    return frame[[ID_COL, TARGET]].rename(columns={TARGET: "prediction"})


def _validate_alignment(reference: pd.DataFrame, frame: pd.DataFrame, name: str) -> None:
    if len(reference) != len(frame):
        raise ValueError(f"{name} row count differs from reference")
    if not np.array_equal(reference[ID_COL].to_numpy(), frame[ID_COL].to_numpy()):
        raise ValueError(f"{name} IDs/order differ from reference")
    if TARGET in reference and TARGET in frame:
        if not np.array_equal(reference[TARGET].to_numpy(), frame[TARGET].to_numpy()):
            raise ValueError(f"{name} target differs from reference")
    if "fold" in reference and "fold" in frame:
        if not np.array_equal(reference["fold"].to_numpy(), frame["fold"].to_numpy()):
            raise ValueError(f"{name} folds differ from reference")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the aligned four-stream live-frontier OOF/test anchor."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--oof-s10-i5", required=True)
    parser.add_argument("--oof-s10-i10", required=True)
    parser.add_argument("--oof-s20-i5", required=True)
    parser.add_argument("--test-lgb-s10-i5", required=True)
    parser.add_argument("--test-xgb-s10-i5", required=True)
    parser.add_argument("--test-lgb-s10-i10", required=True)
    parser.add_argument("--test-lgb-s20-i5", required=True)
    parser.add_argument("--out-dir", default="artifacts/frontier_quad_anchor")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    oof_i5_lgb = _read_oof(args.oof_s10_i5, "lgb")
    oof_i5_xgb = _read_oof(args.oof_s10_i5, "xgb")
    oof_i10_lgb = _read_oof(args.oof_s10_i10, "lgb")
    oof_s20_lgb = _read_oof(args.oof_s20_i5, "prediction")
    oof_frames = [oof_i5_lgb, oof_i5_xgb, oof_i10_lgb, oof_s20_lgb]

    for name, frame in zip(STREAMS, oof_frames):
        if len(frame) != len(train):
            raise ValueError(f"{name} OOF rows do not match train")
        if not np.array_equal(frame[ID_COL].to_numpy(), train[ID_COL].to_numpy()):
            raise ValueError(f"{name} OOF IDs do not match train")
        if not np.array_equal(frame[TARGET].to_numpy(), train[TARGET].to_numpy()):
            raise ValueError(f"{name} OOF target does not match train")
        _validate_alignment(oof_frames[0], frame, name)

    folds = oof_frames[0]["fold"].to_numpy(int)
    unique_folds = sorted(np.unique(folds).tolist())
    if unique_folds != sorted(DEFAULT_ROTATION_WEIGHTS):
        raise ValueError(
            f"expected folds {sorted(DEFAULT_ROTATION_WEIGHTS)}, got {unique_folds}"
        )

    oof_rank_matrix = np.column_stack(
        [_rank01(frame["prediction"]) for frame in oof_frames]
    )
    honest = np.empty(len(train), dtype=float)
    for fold in unique_folds:
        mask = folds == fold
        honest[mask] = oof_rank_matrix[mask] @ DEFAULT_ROTATION_WEIGHTS[fold]
    fixed_mean = oof_rank_matrix @ DEFAULT_MEAN_WEIGHTS
    y = train[TARGET].to_numpy(np.int8)

    test_frames = [
        _read_test(args.test_lgb_s10_i5),
        _read_test(args.test_xgb_s10_i5),
        _read_test(args.test_lgb_s10_i10),
        _read_test(args.test_lgb_s20_i5),
    ]
    for name, frame in zip(STREAMS, test_frames):
        if len(frame) != len(test):
            raise ValueError(f"{name} test rows do not match competition test")
        if not np.array_equal(frame[ID_COL].to_numpy(), test[ID_COL].to_numpy()):
            raise ValueError(f"{name} test IDs do not match competition test")

    test_rank_matrix = np.column_stack(
        [_rank01(frame["prediction"]) for frame in test_frames]
    )
    test_anchor = test_rank_matrix @ DEFAULT_MEAN_WEIGHTS

    oof_out = pd.DataFrame(
        {
            ID_COL: train[ID_COL].to_numpy(),
            TARGET: y,
            "fold": folds,
            "honest_blend": honest,
            "fixed_mean_blend": fixed_mean,
        }
    )
    for index, name in enumerate(STREAMS):
        oof_out[name] = oof_rank_matrix[:, index]
    oof_out.to_csv(out / "oof_anchor.csv", index=False)

    submission_stats = write_submission(
        out / "submission_anchor.csv", sample, test, test_anchor
    )
    report = {
        "version": "live-frontier-quad-anchor-v1",
        "streams": STREAMS,
        "mean_weights": dict(zip(STREAMS, DEFAULT_MEAN_WEIGHTS.tolist())),
        "rotation_weights": {
            str(fold): dict(zip(STREAMS, DEFAULT_ROTATION_WEIGHTS[fold].tolist()))
            for fold in unique_folds
        },
        "honest_oof_auc": float(roc_auc_score(y, honest)),
        "fixed_mean_oof_auc": float(roc_auc_score(y, fixed_mean)),
        "submission": submission_stats,
    }
    (out / "anchor.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
