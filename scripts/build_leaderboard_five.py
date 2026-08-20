#!/usr/bin/env python3
"""Build a five-file Kaggle leaderboard portfolio from the current v19+LGB anchor.

The portfolio deliberately spans a response curve between the current public-anchor
submission and the independently trained LGB identity-screen candidate recovered
from the frontier campaign artifact. This makes five leaderboard slots informative
instead of producing near-duplicate renames.
"""
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
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(frame: pd.DataFrame, expected_ids: np.ndarray, target: str) -> dict:
    if list(frame.columns) != ["id", target]:
        raise ValueError(f"Unexpected columns: {frame.columns.tolist()}; expected ['id', '{target}']")
    if len(frame) != len(expected_ids):
        raise ValueError(f"Row mismatch: {len(frame)} != {len(expected_ids)}")
    ids = frame["id"].to_numpy()
    if not np.array_equal(ids, expected_ids):
        raise ValueError("ID order mismatch")
    pred = frame[target].to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite predictions")
    if len(np.unique(pred)) < 1000:
        raise ValueError("Predictions appear degenerate")
    return {
        "rows": int(len(frame)),
        "unique_predictions": int(len(np.unique(pred))),
        "prediction_min": float(pred.min()),
        "prediction_max": float(pred.max()),
        "prediction_mean": float(pred.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, help="Current v19+LGB public-anchor CSV")
    ap.add_argument("--contrast", required=True, help="Frontier LGB identity-screen candidate CSV")
    ap.add_argument("--out-dir", default="kaggle/submissions_2026-08-19")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    anchor = pd.read_csv(args.anchor)
    contrast = pd.read_csv(args.contrast)
    if len(anchor.columns) != 2 or anchor.columns[0] != "id":
        raise ValueError(f"Anchor schema must be [id, target], got {anchor.columns.tolist()}")
    if len(contrast.columns) != 2 or contrast.columns[0] != "id":
        raise ValueError(f"Contrast schema must be [id, target], got {contrast.columns.tolist()}")

    target = anchor.columns[1]
    ids = anchor["id"].to_numpy()
    if not np.array_equal(ids, contrast["id"].to_numpy()):
        raise ValueError("Anchor and contrast IDs/order differ")

    a = rank01(anchor.iloc[:, 1].to_numpy())
    c = rank01(contrast.iloc[:, 1].to_numpy())
    rank_corr = float(np.corrcoef(a, c)[0, 1])

    # S1 is the current campaign selection. S5 is the independent LGB probe.
    # S2-S4 map the leaderboard response surface between them in rank space.
    recipes = [
        ("01_v19_lgb015_anchor.csv", 1.00, 0.00, "current selected public-anchor + LGB residual"),
        ("02_anchor95_lgb05.csv", 0.95, 0.05, "conservative 5% identity-screen probe"),
        ("03_anchor80_lgb20.csv", 0.80, 0.20, "moderate 20% identity-screen probe"),
        ("04_anchor50_lgb50.csv", 0.50, 0.50, "balanced diversity probe"),
        ("05_lgb_identity_screen.csv", 0.00, 1.00, "independent LGB identity-screen endpoint"),
    ]

    manifest = {
        "purpose": "five Kaggle submissions ordered for immediate leaderboard evaluation",
        "target": target,
        "rows": int(len(ids)),
        "anchor_contrast_rank_correlation": rank_corr,
        "source_anchor": "agent/frontier-097-campaign:kaggle/submission_nomophobia_v19_lgb015.csv",
        "source_contrast_artifact_id": 9346320480,
        "submissions": [],
    }

    for filename, wa, wc, rationale in recipes:
        if wa == 1.0:
            pred = anchor.iloc[:, 1].to_numpy(dtype=float)
        elif wc == 1.0:
            pred = contrast.iloc[:, 1].to_numpy(dtype=float)
        else:
            # Re-rank after blending. For ROC AUC this preserves the intended order
            # while keeping every file on a consistent [0, 1] scale.
            pred = rank01(wa * a + wc * c)
        frame = pd.DataFrame({"id": ids, target: pred})
        stats = validate(frame, ids, target)
        path = out / filename
        frame.to_csv(path, index=False)
        manifest["submissions"].append({
            "order": len(manifest["submissions"]) + 1,
            "file": filename,
            "anchor_weight": wa,
            "contrast_weight": wc,
            "rationale": rationale,
            "sha256": sha256(path),
            **stats,
        })

    # Ensure the five probes really are different in ranking, not cosmetic copies.
    predictions = [pd.read_csv(out / r[0]).iloc[:, 1].to_numpy() for r in recipes]
    corr = np.corrcoef(np.vstack([rank01(x) for x in predictions]))
    manifest["pairwise_rank_correlation"] = corr.tolist()
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            if np.array_equal(predictions[i], predictions[j]):
                raise ValueError(f"Submissions {i+1} and {j+1} are identical")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
