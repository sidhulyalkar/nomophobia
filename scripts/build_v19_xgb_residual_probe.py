#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--weight", type=float, default=0.10)
    ap.add_argument("--out", default="kaggle/next/submission_06_v19_xgb_residual010.csv")
    args = ap.parse_args()

    if not (0.0 <= args.weight <= 0.50):
        raise ValueError("residual weight must be in [0, 0.50]")

    anchor = pd.read_csv(args.anchor)
    if list(anchor.columns) != ["id", "addicted_label"]:
        raise ValueError(f"unexpected anchor schema: {anchor.columns.tolist()}")
    if len(anchor) != 296_302 or anchor["id"].nunique() != len(anchor):
        raise ValueError("anchor row/ID contract failed")

    direction = np.load(args.direction)
    if direction.ndim != 1 or len(direction) != len(anchor):
        raise ValueError("XGB residual direction does not align to test rows")
    if not np.isfinite(direction).all():
        raise ValueError("XGB residual contains non-finite values")

    base = rank01(anchor["addicted_label"].to_numpy(float))
    pred = rank01(base + args.weight * direction)
    if not np.isfinite(pred).all() or len(np.unique(pred)) < 250_000:
        raise ValueError("candidate prediction contract failed")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"id": anchor["id"].to_numpy(), "addicted_label": pred})
    frame.to_csv(out, index=False)

    corr = float(pd.Series(base).corr(pd.Series(pred), method="spearman"))
    report = {
        "candidate": out.name,
        "recipe": "rank(rank(v19_lgb015_anchor) + weight * xgb_identity_screen_treatment_minus_control_rank_direction)",
        "weight": float(args.weight),
        "rows": int(len(frame)),
        "unique_ids": int(frame["id"].nunique()),
        "unique_predictions": int(frame["addicted_label"].nunique()),
        "anchor_candidate_spearman": corr,
        "direction_std": float(np.std(direction)),
        "direction_min": float(np.min(direction)),
        "direction_max": float(np.max(direction)),
        "sha256": sha256(out),
        "evidence_note": (
            "Diagnostic transfer probe. The XGB direction passed matched fixed-schedule OOF residual tests and "
            "received 0.10 deployment weight in the nested quad-anchor winner, but this exact coefficient has "
            "not been selected against v19 OOF."
        ),
    }
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
