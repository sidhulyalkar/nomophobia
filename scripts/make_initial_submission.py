#!/usr/bin/env python
"""Fit the mature dual-view Frontier model on all labeled rows and emit Kaggle submissions.

This script is for *submission generation*, not model selection. The default 1000-tree
counts and 62.5/37.5 rank blend come from repeated S1 experiments; authoritative S3
iteration counts/weights should replace them once available.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s6e8.config import TARGET
from s6e8.features import build_features
from s6e8.io import load_competition
from s6e8.models import make_lgb


def native_categories(train: pd.DataFrame, test: pd.DataFrame):
    """Prepare only the native categorical frames needed by LightGBM."""
    tr = train.copy(); te = test.copy(); cats = []
    for c in tr.columns:
        if tr[c].dtype == "object" or str(tr[c].dtype).startswith(("string", "category")):
            cats.append(c)
            a = tr[c].astype("string").fillna("MISSING").astype(str)
            b = te[c].astype("string").fillna("MISSING").astype(str)
            vocab = sorted(set(a.unique()).union(b.unique()))
            dt = pd.CategoricalDtype(vocab, ordered=False)
            tr[c] = a.astype(dt); te[c] = b.astype(dt)
    return tr, te, cats


def unit_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (rankdata(x, method="average") - 0.5) / len(x)


def fit_predict_lgb(X, y, T, cats, *, seed, estimators, profile, model_path: Path | None):
    t0 = time.time(); model = make_lgb(seed, estimators, profile, device="cpu")
    model.fit(X, y, categorical_feature=cats)
    pred = model.predict_proba(T)[:, 1]; elapsed = time.time() - t0
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(model_path))
    return pred, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="artifacts/submissions/initial_v0_1")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--combined-estimators", type=int, default=1000)
    ap.add_argument("--raw-estimators", type=int, default=1000)
    ap.add_argument("--raw-weight", type=float, default=0.375)
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train, test, sample = load_competition(args.data_dir)
    y = train[TARGET].astype(int).to_numpy(); train_x = train.drop(columns=[TARGET])
    print(f"train={len(train):,} test={len(test):,} positive={y.mean():.6f}", flush=True)

    t0 = time.time()
    Xc, Tc = build_features(train_x, test, use_frequency=True, frequency_reference=(train_x, test))
    print(f"combined_features={Xc.shape[1]} feature_seconds={time.time()-t0:.1f}", flush=True)
    n_features = int(Xc.shape[1])
    Xc, Tc, ccats = native_categories(Xc, Tc)
    pc, sec_c = fit_predict_lgb(
        Xc, y, Tc, ccats, seed=args.seed, estimators=args.combined_estimators,
        profile="combined63", model_path=(out / "models" / "combined63.txt") if args.save_models else None,
    )
    np.save(out / "test_combined_probability.npy", pc)
    print(f"combined_fit_predict_seconds={sec_c:.1f}", flush=True)
    del Xc, Tc; gc.collect()

    Xr = train_x.drop(columns=["id"], errors="ignore").copy()
    Tr = test.drop(columns=["id"], errors="ignore").copy()
    Xr, Tr, rcats = native_categories(Xr, Tr)
    pr, sec_r = fit_predict_lgb(
        Xr, y, Tr, rcats, seed=args.seed + 101, estimators=args.raw_estimators,
        profile="raw63", model_path=(out / "models" / "raw63.txt") if args.save_models else None,
    )
    np.save(out / "test_raw_probability.npy", pr)
    print(f"raw_fit_predict_seconds={sec_r:.1f}", flush=True)

    rc = unit_rank(pc); rr = unit_rank(pr)
    weights = sorted(set([0.0, args.raw_weight, 0.325, 0.375, 0.425]))
    submissions = {}
    for w in weights:
        score = (1.0 - w) * rc + w * rr
        label = f"dualview_raw{int(round(w*1000)):03d}" if w > 0 else "combined_only"
        path = out / f"submission_{label}.csv"
        sub = sample.copy(); sub[TARGET] = score; sub.to_csv(path, index=False)
        submissions[label] = str(path.name)

    primary = out / "submission.csv"
    primary_score = (1.0 - args.raw_weight) * rc + args.raw_weight * rr
    sub = sample.copy(); sub[TARGET] = primary_score; sub.to_csv(primary, index=False)

    meta = {
        "version": "nomophobia-v0.1-initial",
        "scientific_status": "initial submission candidate; S1-derived configuration, not S3-promoted",
        "train_rows": int(len(train)), "test_rows": int(len(test)), "seed": args.seed,
        "combined_estimators": args.combined_estimators, "raw_estimators": args.raw_estimators,
        "primary_raw_weight": args.raw_weight, "feature_count_combined": n_features,
        "fit_seconds": {"combined": sec_c, "raw": sec_r}, "portfolio": submissions,
        "primary": primary.name,
        "prediction_correlation_probability": float(np.corrcoef(pc, pr)[0, 1]),
        "prediction_correlation_rank": float(np.corrcoef(rc, rr)[0, 1]),
        "submission_min": float(primary_score.min()), "submission_max": float(primary_score.max()),
        "submission_nan": int(np.isnan(primary_score).sum()),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
