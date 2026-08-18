#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from live_frontier_candidate import CAT, TARGET, feature_frame, fold_te


def fit_model(x: pd.DataFrame, y: np.ndarray, seed: int, estimators: int):
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=estimators,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2.0,
        max_bin=255,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y, categorical_feature=CAT)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="innerfold_screen.json")
    ap.add_argument("--rows", type=int, default=120000)
    ap.add_argument("--estimators", type=int, default=1200)
    ap.add_argument("--smoothing", type=float, default=10.0)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.data_dir) / "train.csv")
    y_all = df[TARGET].to_numpy(np.int8)
    for c in CAT:
        df[c] = pd.Categorical(df[c], categories=pd.Index(df[c].dropna().unique()))

    rows = []
    for seed in [20260816, 20260817, 20260818]:
        idx, _ = train_test_split(
            np.arange(len(df)),
            train_size=args.rows,
            stratify=y_all,
            random_state=seed,
        )
        y = y_all[idx]
        ti, vi = train_test_split(
            np.arange(len(idx)),
            test_size=0.25,
            stratify=y,
            random_state=seed + 91,
        )
        tr = df.iloc[idx[ti]].reset_index(drop=True)
        va = df.iloc[idx[vi]].reset_index(drop=True)
        yt = y[ti]
        yv = y[vi]

        e5t, e5v, _ = fold_te(
            tr, yt, va, va.iloc[:1].copy(), 5, args.smoothing, seed + 777
        )
        e10t, e10v, _ = fold_te(
            tr, yt, va, va.iloc[:1].copy(), 10, args.smoothing, seed + 777
        )
        m5 = fit_model(feature_frame(tr, e5t, False), yt, seed, args.estimators)
        m10 = fit_model(
            feature_frame(tr, e10t, False), yt, seed + 1000, args.estimators
        )
        p5 = m5.predict_proba(feature_frame(va, e5v, False))[:, 1]
        p10 = m10.predict_proba(feature_frame(va, e10v, False))[:, 1]
        a5 = float(roc_auc_score(yv, p5))
        a10 = float(roc_auc_score(yv, p10))
        row = {
            "seed": seed,
            "inner5_auc": a5,
            "inner10_auc": a10,
            "delta": a10 - a5,
            "rank_corr": float(pd.Series(p5).rank().corr(pd.Series(p10).rank())),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    deltas = np.array([r["delta"] for r in rows])
    summary = {
        "results": rows,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "positive_seeds": int((deltas > 0).sum()),
        "decision": (
            "ADVANCE_INNER10"
            if (deltas > 0).sum() >= 2 and deltas.mean() > 0
            else "KEEP_INNER5"
        ),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
