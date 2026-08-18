#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "addicted_label"
ID = "id"
CAT = ["gender", "stress_level", "academic_work_impact"]
NUM = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
RAW = NUM + CAT
MISS = -1_000_000.0


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True).to_numpy(np.float64)


def smoothed_map(values: pd.Series, y: np.ndarray, smoothing: float) -> pd.Series:
    prior = float(np.mean(y))
    key = values.astype("float64").fillna(MISS)
    frame = pd.DataFrame({"key": key.to_numpy(), "y": y})
    stat = frame.groupby("key", sort=False, observed=True)["y"].agg(["sum", "count"])
    return (stat["sum"] + smoothing * prior) / (stat["count"] + smoothing)


def apply_map(values: pd.Series, mapping: pd.Series, prior: float) -> np.ndarray:
    return (
        values.astype("float64")
        .fillna(MISS)
        .map(mapping)
        .fillna(prior)
        .to_numpy(np.float32)
    )


def fold_te(
    train: pd.DataFrame,
    y: np.ndarray,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    inner_folds: int,
    smoothing: float,
    seed: int,
):
    prior = float(np.mean(y))
    etr = pd.DataFrame(index=np.arange(len(train)))
    eva = pd.DataFrame(index=np.arange(len(valid)))
    ete = pd.DataFrame(index=np.arange(len(test)))
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    for col in NUM:
        oof = np.empty(len(train), np.float32)
        for ti, vi in inner.split(train, y):
            mp = smoothed_map(train.iloc[ti][col], y[ti], smoothing)
            oof[vi] = apply_map(train.iloc[vi][col], mp, float(np.mean(y[ti])))
        full = smoothed_map(train[col], y, smoothing)
        etr[f"te__{col}"] = oof
        eva[f"te__{col}"] = apply_map(valid[col], full, prior)
        ete[f"te__{col}"] = apply_map(test[col], full, prior)
    return etr, eva, ete


def feature_frame(raw: pd.DataFrame, te: pd.DataFrame, xgb_mode: bool) -> pd.DataFrame:
    out = raw[RAW].copy().reset_index(drop=True)
    daily = raw["daily_screen_time_hours"].reset_index(drop=True).replace(0, np.nan)
    out["parts_sum"] = raw[
        ["social_media_hours", "gaming_hours", "work_study_hours"]
    ].reset_index(drop=True).sum(axis=1)
    out["social_media_share"] = raw["social_media_hours"].reset_index(drop=True) / daily
    out["gaming_share"] = raw["gaming_hours"].reset_index(drop=True) / daily
    out["work_study_share"] = raw["work_study_hours"].reset_index(drop=True) / daily
    out["weekend_minus_daily"] = (
        raw["weekend_screen_time"].reset_index(drop=True) - daily
    )
    for c in te.columns:
        out[c] = te[c].to_numpy()
    if xgb_mode:
        for c in CAT:
            out[c] = out[c].cat.codes.replace(-1, np.nan).astype(np.float32)
        out = out.astype(np.float32)
    return out


def best_rotating_blend(y, folds, pred_lgb, pred_xgb):
    rl, rx = rank01(pred_lgb), rank01(pred_xgb)
    weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    honest = np.empty(len(y), np.float64)
    chosen = []
    for held in sorted(np.unique(folds)):
        sel = folds != held
        val = folds == held
        scores = [
            roc_auc_score(y[sel], w * rl[sel] + (1 - w) * rx[sel])
            for w in weights
        ]
        w = float(weights[int(np.argmax(scores))])
        chosen.append(w)
        honest[val] = w * rl[val] + (1 - w) * rx[val]
    mean_w = float(np.mean(chosen))
    return honest, chosen, mean_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="live_results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--smoothing", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--lgb-estimators", type=int, default=4500)
    ap.add_argument("--xgb-estimators", type=int, default=3000)
    a = ap.parse_args()

    started = time.time()
    data = Path(a.data_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    if len(train) != 691369 or len(test) != 296302:
        raise ValueError(f"unexpected shape train={train.shape} test={test.shape}")
    if not sample[ID].equals(test[ID]):
        raise ValueError("sample IDs do not match test IDs")
    expected_train = {ID, TARGET, *RAW}
    expected_test = {ID, *RAW}
    if set(train.columns) != expected_train or set(test.columns) != expected_test:
        raise ValueError("competition schema mismatch")

    for c in CAT:
        cats = pd.Index(
            pd.concat([train[c], test[c]], ignore_index=True).dropna().unique()
        )
        train[c] = pd.Categorical(train[c], categories=cats)
        test[c] = pd.Categorical(test[c], categories=cats)

    y = train[TARGET].to_numpy(np.int8)
    outer = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
    fold_id = np.empty(len(train), np.int8)
    oof_lgb = np.empty(len(train), np.float64)
    oof_xgb = np.empty(len(train), np.float64)
    test_lgb = np.zeros(len(test), np.float64)
    test_xgb = np.zeros(len(test), np.float64)
    fold_rows = []

    for f, (ti, vi) in enumerate(outer.split(train, y), 0):
        fold_id[vi] = f
        t0 = time.time()
        tr = train.iloc[ti].reset_index(drop=True)
        va = train.iloc[vi].reset_index(drop=True)
        yy = y[ti]
        yv = y[vi]
        etr, eva, ete = fold_te(
            tr,
            yy,
            va,
            test.reset_index(drop=True),
            a.inner_folds,
            a.smoothing,
            a.seed + 100 + f,
        )
        xl_tr = feature_frame(tr, etr, False)
        xl_va = feature_frame(va, eva, False)
        xl_te = feature_frame(test, ete, False)

        ml = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=a.lgb_estimators,
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
            random_state=a.seed + f,
            n_jobs=-1,
            verbosity=-1,
        )
        ml.fit(
            xl_tr,
            yy,
            eval_set=[(xl_va, yv)],
            eval_metric="auc",
            categorical_feature=CAT,
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        pl = ml.predict_proba(xl_va, num_iteration=ml.best_iteration_)[:, 1]
        tl = ml.predict_proba(xl_te, num_iteration=ml.best_iteration_)[:, 1]
        oof_lgb[vi] = pl
        test_lgb += tl / a.folds

        xx_tr = feature_frame(tr, etr, True)
        xx_va = feature_frame(va, eva, True)
        xx_te = feature_frame(test, ete, True)
        mx = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=a.xgb_estimators,
            learning_rate=0.035,
            max_depth=8,
            min_child_weight=20.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=2.0,
            max_bin=256,
            tree_method="hist",
            random_state=a.seed + f,
            n_jobs=-1,
            early_stopping_rounds=150,
        )
        mx.fit(xx_tr, yy, eval_set=[(xx_va, yv)], verbose=False)
        px = mx.predict_proba(xx_va)[:, 1]
        tx = mx.predict_proba(xx_te)[:, 1]
        oof_xgb[vi] = px
        test_xgb += tx / a.folds
        fold_rows.append(
            {
                "fold": f,
                "lgb_auc": float(roc_auc_score(yv, pl)),
                "xgb_auc": float(roc_auc_score(yv, px)),
                "lgb_best_iteration": int(ml.best_iteration_),
                "xgb_best_iteration": int(mx.best_iteration),
                "seconds": round(time.time() - t0, 2),
            }
        )
        print(json.dumps(fold_rows[-1]), flush=True)

    honest, held_weights_lgb, mean_w_lgb = best_rotating_blend(
        y, fold_id, oof_lgb, oof_xgb
    )
    auc_lgb = float(roc_auc_score(y, oof_lgb))
    auc_xgb = float(roc_auc_score(y, oof_xgb))
    auc_blend = float(roc_auc_score(y, honest))
    corr = float(pd.Series(oof_lgb).rank().corr(pd.Series(oof_xgb).rank()))

    rl_test, rx_test = rank01(test_lgb), rank01(test_xgb)
    test_blend = mean_w_lgb * rl_test + (1 - mean_w_lgb) * rx_test
    candidates = {"lgb": auc_lgb, "xgb": auc_xgb, "honest_blend": auc_blend}
    selected = max(candidates, key=candidates.get)
    if selected == "lgb":
        final = test_lgb
    elif selected == "xgb":
        final = test_xgb
    else:
        final = test_blend

    for name, pred in (
        ("lgb", test_lgb),
        ("xgb", test_xgb),
        ("blend", test_blend),
        ("next", final),
    ):
        sub = sample[[ID]].copy()
        sub[TARGET] = pred
        sub.to_csv(out / f"submission_{name}.csv", index=False)
    np.save(out / "oof_lgb.npy", oof_lgb)
    np.save(out / "oof_xgb.npy", oof_xgb)
    np.save(out / "test_lgb.npy", test_lgb)
    np.save(out / "test_xgb.npy", test_xgb)
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_id,
            "lgb": oof_lgb,
            "xgb": oof_xgb,
            "honest_blend": honest,
        }
    ).to_csv(out / "oof.csv", index=False)
    decision = {
        "version": "live-frontier-v1",
        "rows": {"train": len(train), "test": len(test)},
        "folds": a.folds,
        "inner_folds": a.inner_folds,
        "smoothing": a.smoothing,
        "auc": candidates,
        "fold_metrics": fold_rows,
        "rank_correlation_lgb_xgb": corr,
        "rotating_lgb_weights": held_weights_lgb,
        "mean_lgb_weight_for_test": mean_w_lgb,
        "selected": selected,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
