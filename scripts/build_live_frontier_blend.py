#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID = "id"
TARGET = "addicted_label"
EXPECTED_TEST_ROWS = 296302


def rank01(values: pd.Series) -> np.ndarray:
    return values.rank(method="average", pct=True).to_numpy(np.float64)


def validate_submission(frame: pd.DataFrame) -> None:
    if list(frame.columns) != [ID, TARGET]:
        raise ValueError(f"submission columns must be {[ID, TARGET]}")
    if len(frame) != EXPECTED_TEST_ROWS:
        raise ValueError(f"expected {EXPECTED_TEST_ROWS} rows, got {len(frame)}")
    if not frame[ID].is_unique:
        raise ValueError("submission IDs must be unique")
    if not np.isfinite(frame[TARGET].to_numpy(float)).all():
        raise ValueError("submission contains non-finite predictions")
    if frame[TARGET].nunique() < 2:
        raise ValueError("submission predictions are constant")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a deterministic percentile-rank blend from two test submissions."
    )
    ap.add_argument("--lgb", required=True, help="inner-10 LightGBM submission CSV")
    ap.add_argument("--xgb", required=True, help="XGBoost submission CSV")
    ap.add_argument("--lgb-weight", type=float, default=0.46)
    ap.add_argument("--out", default="submission_nomophobia_frontier_i10_xgb_v1.csv")
    args = ap.parse_args()

    if not 0.0 <= args.lgb_weight <= 1.0:
        raise ValueError("--lgb-weight must be in [0, 1]")

    lgb = pd.read_csv(args.lgb)
    xgb = pd.read_csv(args.xgb)
    validate_submission(lgb)
    validate_submission(xgb)
    if not lgb[ID].equals(xgb[ID]):
        raise ValueError("LGB and XGB submission IDs/order differ")

    blend = pd.DataFrame(
        {
            ID: lgb[ID].to_numpy(),
            TARGET: args.lgb_weight * rank01(lgb[TARGET])
            + (1.0 - args.lgb_weight) * rank01(xgb[TARGET]),
        }
    )
    validate_submission(blend)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blend.to_csv(out, index=False)
    round_trip = pd.read_csv(out)
    validate_submission(round_trip)
    if not round_trip[ID].equals(blend[ID]):
        raise ValueError("serialized submission changed ID order")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {
        "file": out.name,
        "method": "percentile-rank blend",
        "lgb_weight": args.lgb_weight,
        "xgb_weight": 1.0 - args.lgb_weight,
        "rows": len(round_trip),
        "sha256": digest,
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
