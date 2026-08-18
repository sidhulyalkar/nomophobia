#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from live_frontier_candidate import CAT, MISS, NUM, RAW, TARGET, feature_frame


def stats_map(values: pd.Series, y: np.ndarray, smoothing: float):
    prior = float(np.mean(y))
    key = values.astype("float64").fillna(MISS)
    frame = pd.DataFrame({"key": key.to_numpy(), "y": y})
    stat = frame.groupby("key", sort=False, observed=True)["y"].agg(["sum", "count"])
    mean = (stat["sum"] + smoothing * prior) / (stat["count"] + smoothing)
    return pd.DataFrame({"mean": mean, "count": stat["count"].astype(float)})


def apply_stats(values: pd.Series, stats: pd.DataFrame, prior: float, smoothing: float):
    key = values.astype("float64").fillna(MISS)
    mean = key.map(stats["mean"]).fillna(prior).to_numpy(np.float32)
    count = key.map(stats["count"]).fillna(0.0).to_numpy(np.float32)
    support_log = np.log1p(count).astype(np.float32)
    reliability = (count / (count + smoothing)).astype(np.float32)
    denom = np.sqrt(max(prior * (1.0 - prior), 1e-6))
    evidence_z = ((mean - prior) * np.sqrt(count + smoothing) / denom).astype(np.float32)
    return mean, support_log, reliability, evidence_z


def encode(
    train: pd.DataFrame,
    y: np.ndarray,
    valid: pd.DataFrame,
    inner_folds: int,
    smoothing: float,
    seed: int,
):
    prior = float(np.mean(y))
    base_tr = pd.DataFrame(index=np.arange(len(train)))
    base_va = pd.DataFrame(index=np.arange(len(valid)))
    ext_tr = pd.DataFrame(index=np.arange(len(train)))
    ext_va = pd.DataFrame(index=np.arange(len(valid)))
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)

    for col in NUM:
        tr_mean = np.empty(len(train), np.float32)
        tr_support = np.empty(len(train), np.float32)
        tr_rel = np.empty(len(train), np.float32)
        tr_z = np.empty(len(train), np.float32)
        for ti, vi in inner.split(train, y):
            stat = stats_map(train.iloc[ti][col], y[ti], smoothing)
            vals = apply_stats(
                train.iloc[vi][col], stat, float(np.mean(y[ti])), smoothing
            )
            tr_mean[vi], tr_support[vi], tr_rel[vi], tr_z[vi] = vals

        full = stats_map(train[col], y, smoothing)
        va_mean, va_support, va_rel, va_z = apply_stats(
            valid[col], full, prior, smoothing
        )
        base_tr[f"te__{col}"] = tr_mean
        base_va[f"te__{col}"] = va_mean
        ext_tr[f"te_support_log__{col}"] = tr_support
        ext_va[f"te_support_log__{col}"] = va_support
        ext_tr[f"te_reliability__{col}"] = tr_rel
        ext_va[f"te_reliability__{col}"] = va_rel
        ext_tr[f"te_evidence_z__{col}"] = tr_z
        ext_va[f"te_evidence_z__{col}"] = va_z

    return base_tr, base_va, ext_tr, ext_va


def with_extra(raw: pd.DataFrame, te: pd.DataFrame, extra: pd.DataFrame | None):
    out = feature_frame(raw, te, False)
    if extra is not None:
        for c in extra.columns:
            out[c] = extra[c].to_numpy()
    return out


def model(seed: int, estimators: int):
    return lgb.LGBMClassifier(
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="te_uncertainty_screen.json")
    ap.add_argument("--rows", type=int, default=120000)
    ap.add_argument("--estimators", type=int, default=1200)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--smoothing", type=float, default=10.0)
    args = ap.parse_args()

    train = pd.read_csv(Path(args.data_dir) / "train.csv")
    if len(train) != 691369:
        raise ValueError(f"unexpected train rows {len(train)}")
    if set(train.columns) != {"id", TARGET, *RAW}:
        raise ValueError("schema mismatch")
    for c in CAT:
        train[c] = pd.Categorical(
            train[c], categories=pd.Index(train[c].dropna().unique())
        )

    y_all = train[TARGET].to_numpy(np.int8)
    results = []
    for seed in [20260816, 20260817, 20260818]:
        idx, _ = train_test_split(
            np.arange(len(train)),
            train_size=args.rows,
            stratify=y_all,
            random_state=seed,
        )
        y = y_all[idx]
        tr_idx, va_idx = train_test_split(
            np.arange(len(idx)),
            test_size=0.25,
            stratify=y,
            random_state=seed + 91,
        )
        tr = train.iloc[idx[tr_idx]].reset_index(drop=True)
        va = train.iloc[idx[va_idx]].reset_index(drop=True)
        yy = y[tr_idx]
        yv = y[va_idx]

        te_tr, te_va, ex_tr, ex_va = encode(
            tr, yy, va, args.inner_folds, args.smoothing, seed + 777
        )
        xb_tr = with_extra(tr, te_tr, None)
        xb_va = with_extra(va, te_va, None)
        xu_tr = with_extra(tr, te_tr, ex_tr)
        xu_va = with_extra(va, te_va, ex_va)

        mb = model(seed, args.estimators)
        mu = model(seed + 1000, args.estimators)
        mb.fit(xb_tr, yy, categorical_feature=CAT)
        mu.fit(xu_tr, yy, categorical_feature=CAT)
        pb = mb.predict_proba(xb_va)[:, 1]
        pu = mu.predict_proba(xu_va)[:, 1]
        ab = float(roc_auc_score(yv, pb))
        au = float(roc_auc_score(yv, pu))
        row = {
            "seed": seed,
            "baseline_auc": ab,
            "uncertainty_auc": au,
            "delta": au - ab,
            "rank_corr": float(pd.Series(pb).rank().corr(pd.Series(pu).rank())),
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    deltas = np.array([r["delta"] for r in results], dtype=float)
    summary = {
        "hypothesis": "leakage-safe support/reliability/evidence transforms improve exact-value TE",
        "rows_per_seed": args.rows,
        "validation_fraction": 0.25,
        "estimators": args.estimators,
        "inner_folds": args.inner_folds,
        "smoothing": args.smoothing,
        "results": results,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "positive_seeds": int((deltas > 0).sum()),
        "decision": (
            "ADVANCE"
            if (deltas > 0).sum() >= 2 and deltas.mean() > 0
            else "KILL"
        ),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
