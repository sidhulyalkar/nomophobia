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
DEFAULT_WEIGHTS = {
    "lgb_s10_i5": 0.15,
    "xgb_s10_i5": 0.44,
    "lgb_s10_i10": 0.26,
    "lgb_s20_i5": 0.15,
}


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
        description="Build the validated four-stream NOMOPHOBIA live-frontier rank blend."
    )
    ap.add_argument("--lgb-s10-i5", required=True)
    ap.add_argument("--xgb-s10-i5", required=True)
    ap.add_argument("--lgb-s10-i10", required=True)
    ap.add_argument("--lgb-s20-i5", required=True)
    ap.add_argument("--w-lgb-s10-i5", type=float, default=DEFAULT_WEIGHTS["lgb_s10_i5"])
    ap.add_argument("--w-xgb-s10-i5", type=float, default=DEFAULT_WEIGHTS["xgb_s10_i5"])
    ap.add_argument("--w-lgb-s10-i10", type=float, default=DEFAULT_WEIGHTS["lgb_s10_i10"])
    ap.add_argument("--w-lgb-s20-i5", type=float, default=DEFAULT_WEIGHTS["lgb_s20_i5"])
    ap.add_argument("--out", default="submission_nomophobia_frontier_quad_v2.csv")
    args = ap.parse_args()

    paths = {
        "lgb_s10_i5": args.lgb_s10_i5,
        "xgb_s10_i5": args.xgb_s10_i5,
        "lgb_s10_i10": args.lgb_s10_i10,
        "lgb_s20_i5": args.lgb_s20_i5,
    }
    weights = {
        "lgb_s10_i5": args.w_lgb_s10_i5,
        "xgb_s10_i5": args.w_xgb_s10_i5,
        "lgb_s10_i10": args.w_lgb_s10_i10,
        "lgb_s20_i5": args.w_lgb_s20_i5,
    }
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("blend weights must be non-negative")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
        raise ValueError(f"blend weights must sum to 1.0, got {sum(weights.values())}")

    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    base = frames["xgb_s10_i5"]
    validate_submission(base)
    for name, frame in frames.items():
        validate_submission(frame)
        if not frame[ID].equals(base[ID]):
            raise ValueError(f"{name} IDs/order differ from XGBoost stream")

    prediction = np.zeros(len(base), dtype=np.float64)
    for name, weight in weights.items():
        prediction += weight * rank01(frames[name][TARGET])

    blend = pd.DataFrame({ID: base[ID].to_numpy(), TARGET: prediction})
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
        "weights": weights,
        "rows": len(round_trip),
        "sha256": digest,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
