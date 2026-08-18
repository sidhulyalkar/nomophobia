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

TARGET = "addicted_label"
ID = "id"
CAT = ["gender", "stress_level", "academic_work_impact"]
NUM = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
RAW = NUM + CAT
MISS = -1_000_000.0


def stats_map(values: pd.Series, y: np.ndarray, smoothing: float):
    prior = float(np.mean(y))
    key = values.astype("float64").fillna(MISS)
    f = pd.DataFrame({"key": key.to_numpy(), "y": y})
    st = f.groupby("key", sort=False, observed=True)["y"].agg(["sum", "count"])
    mean = (st["sum"] + smoothing * prior) / (st["count"] + smoothing)
    return pd.DataFrame({"mean": mean, "count": st["count"].astype(float)})


def apply_stats(values: pd.Series, stats: pd.DataFrame, prior: float, smoothing: float):
    key = values.astype("float64").fillna(MISS)
    mean = key.map(stats["mean"]).fillna(prior).to_numpy(np.float32)
    count = key.map(stats["count"]).fillna(0.0).to_numpy(np.float32)
    support_log = np.log1p(count).astype(np.float32)
    reliability = (count / (count + smoothing)).astype(np.float32)
    denom = np.sqrt(np.maximum(prior * (1.0 - prior), 1e-6))
    evidence_z = ((mean - prior) * np.sqrt(count + smoothing) / denom).astype(np.float32)
    return mean, support_log, reliability, evidence_z


def encode(train: pd.DataFrame, y: np.ndarray, valid: pd.DataFrame, inner_folds: int,
           smoothing: float, seed: int):
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
            st = stats_map(train.iloc[ti][col], y[ti], smoothing)
            vals = apply_stats(train.iloc[vi][col], st, float(np.mean(y[ti])), smoothing)
            tr_mean[vi], tr_support[vi], tr_rel[vi], tr_z[vi] = vals

        full = stats_map(train[col], y, smoothing)
        va_mean, va_support, va_rel, va_z = apply_stats(valid[col], full, prior, smoothing)

        base_tr[f"te__{col}"] = tr_mean
        base_va[f"te__{col}"] = va_mean
        ext_tr[f"te_support_log__{col}"] = tr_support
        ext_va[f"te_support_log__{col}"] = va_support
        ext_tr[f"te_reliability__{col}"] = tr_rel
        ext_va[f"te_reliability__{col}"] = va_rel
        ext_tr[f"te_evidence_z__{col}"] = tr_z
        ext_va[f"te_evidence_z__{col}"] = va_z

    return base_tr, base_va, ext_tr, ext_va


def frame(raw: pd.DataFrame, te: pd.DataFrame, extra: pd.DataFrame | None = None):
    out = raw[RAW].copy().reset_index(drop=True)
    daily = raw["daily_screen_time_hours"].reset_index(drop=True).replace(0, np.nan)
    out["parts_sum"] = raw[["social_media_hours", "gaming_hours", "work_study_hours"]].reset_index(drop=True).sum(axis=1)
    out["social_media_share"] = raw["social_media_hours"].reset_index(drop=True) / daily
    out["gaming_share"] = raw["gaming_hours"].reset_index(drop=True) / daily
    out["work_study_share"] = raw["work_study_hours"].reset_index(drop=True) / daily
    out["weekend_minus_daily"] = raw["weekend_screen_time"].reset_index(drop=True) - daily
    for c in te.columns:
        out[c] = te[c].to_numpy()
    if extra is not None:
        for c in extra.columns:
            out[c] = extra[c].to_numpy()
    return out


def model(seed: int, estimators: int):
    return lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=estimators,
        learning_rate=0.035, num_leaves=31, max_depth=-1,
        min_child_samples=100, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=2.0,
        max_bin=255, random_state=seed, n_jobs=-1, verbosity=-1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="te_uncertainty_screen.json")
    ap.add_argument("--rows", type=int, default=120000)
    ap.add_argument("--estimators", type=int, default=1200)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--smoothing", type=float, default=10.0)
    a = ap.parse_args()

    train = pd.read_csv(Path(a.data_dir) / "train.csv")
    test = pd.read_csv(Path(a.data_dir) / "test.csv", nrows=5)
    if len(train) != 691369:
        raise ValueError(f"unexpected train rows {len(train)}")
    expected = {ID, TARGET, *RAW}
    if set(train.columns) != expected:
        raise ValueError("schema mismatch")

    for c in CAT:
        cats = pd.Index(train[c].dropna().unique())
        train[c] = pd.Categorical(train[c], categories=cats)

    y_all = train[TARGET].to_numpy(np.int8)
    seeds = [20260816, 20260817, 20260818]
    results = []

    for seed in seeds:
        idx, _ = train_test_split(
            np.arange(len(train)), train_size=a.rows, stratify=y_all, random_state=seed
        )
        y = y_all[idx]
        tr_idx, va_idx = train_test_split(
            np.arange(len(idx)), test_size=0.25, stratify=y, random_state=seed + 91
        )
        tr = train.iloc[idx[tr_idx]].reset_index(drop=True)
        va = train.iloc[idx[va_idx]].reset_index(drop=True)
        yy = y[tr_idx]
        yv = y[va_idx]

        te_tr, te_va, ex_tr, ex_va = encode(
            tr, yy, va, a.inner_folds, a.smoothing, seed + 777
        )
        xb_tr = frame(tr, te_tr)
        xb_va = frame(va, te_va)
        xu_tr = frame(tr, te_tr, ex_tr)
        xu_va = frame(va, te_va, ex_va)

        mb = model(seed, a.estimators)
        mu = model(seed, a.estimators)
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
        "rows_per_seed": a.rows,
        "validation_fraction": 0.25,
        "estimators": a.estimators,
        "inner_folds": a.inner_folds,
        "smoothing": a.smoothing,
        "results": results,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "positive_seeds": int((deltas > 0).sum()),
        "decision": "ADVANCE" if (deltas > 0).sum() >= 2 and deltas.mean() > 0 else "KILL",
    }
    Path(a.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
