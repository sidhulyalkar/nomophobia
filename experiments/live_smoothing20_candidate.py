#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from live_frontier_candidate import CAT, ID, TARGET, feature_frame, fold_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="s20_results")
    ap.add_argument("--estimators", type=int, default=4500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260801)
    args = ap.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    if len(train) != 691369 or len(test) != 296302:
        raise ValueError("competition row-count contract mismatch")
    if not test[ID].equals(sample[ID]):
        raise ValueError("sample IDs do not match test IDs")

    for c in CAT:
        cats = pd.Index(
            pd.concat([train[c], test[c]], ignore_index=True).dropna().unique()
        )
        train[c] = pd.Categorical(train[c], categories=cats)
        test[c] = pd.Categorical(test[c], categories=cats)

    y = train[TARGET].to_numpy(np.int8)
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train), np.float64)
    test_pred = np.zeros(len(test), np.float64)
    fold = np.zeros(len(train), np.int8)
    metrics = []
    started = time.time()

    for f, (ti, vi) in enumerate(cv.split(train, y)):
        tr = train.iloc[ti].reset_index(drop=True)
        va = train.iloc[vi].reset_index(drop=True)
        yt = y[ti]
        yv = y[vi]
        etr, eva, ete = fold_te(
            tr,
            yt,
            va,
            test.reset_index(drop=True),
            args.inner_folds,
            20.0,
            args.seed + 100 + f,
        )
        xt = feature_frame(tr, etr, False)
        xv = feature_frame(va, eva, False)
        xte = feature_frame(test, ete, False)
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=args.estimators,
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
            random_state=args.seed + f,
            n_jobs=-1,
            verbosity=-1,
        )
        t0 = time.time()
        model.fit(
            xt,
            yt,
            eval_set=[(xv, yv)],
            eval_metric="auc",
            categorical_feature=CAT,
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        pv = model.predict_proba(xv, num_iteration=model.best_iteration_)[:, 1]
        pt = model.predict_proba(xte, num_iteration=model.best_iteration_)[:, 1]
        oof[vi] = pv
        test_pred += pt / args.folds
        fold[vi] = f
        row = {
            "fold": f,
            "auc": float(roc_auc_score(yv, pv)),
            "best_iteration": int(model.best_iteration_),
            "seconds": round(time.time() - t0, 2),
        }
        metrics.append(row)
        print(json.dumps(row), flush=True)

    auc = float(roc_auc_score(y, oof))
    np.save(out / "oof_s20.npy", oof)
    np.save(out / "test_s20.npy", test_pred)
    np.save(out / "fold.npy", fold)
    np.save(out / "target.npy", y)
    pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(
        out / "submission_s20.csv", index=False
    )
    pd.DataFrame({ID: train[ID], TARGET: y, "fold": fold, "prediction": oof}).to_csv(
        out / "oof_s20.csv", index=False
    )
    decision = {
        "rows": {"train": len(train), "test": len(test)},
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "smoothing": 20.0,
        "auc": auc,
        "fold_metrics": metrics,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
