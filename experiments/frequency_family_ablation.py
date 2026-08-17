#!/usr/bin/env python
"""Mature-capacity decomposition of the load-bearing frequency feature family.

This experiment does not promote a new model. It asks which frequency subfamily carries
most of the observed gain while keeping the behavioral/digit backbone and model capacity
fixed. Frequency maps are target-free and use the full train+test predictor population,
matching the project's transductive deployment setting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s6e8.artifacts import atomic_write_json
from s6e8.config import TARGET
from s6e8.cv import frozen_folds
from s6e8.features import build_features
from s6e8.frequency import FREQUENCY_ARMS, frequency_columns_for_arm, frequency_feature_groups
from s6e8.io import load_competition
from s6e8.models import make_lgb
from s6e8.preprocess import prepare_tree_frames


def _sample_rows(y: np.ndarray, rows: int, seed: int) -> np.ndarray:
    if rows >= len(y):
        return np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=rows, random_state=seed)
    idx, _ = next(splitter.split(np.zeros(len(y)), y))
    return np.sort(idx)


def _evaluate_arm(X, y, folds, *, estimators: int, seed: int, device: str) -> dict:
    _, _, native, _, cats = prepare_tree_frames(X, X.iloc[:1].copy())
    oof = np.empty(len(y), dtype=float)
    fold_auc = []
    for fold in np.unique(folds):
        train_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        model = make_lgb(
            seed + 1009 * int(fold),
            estimators,
            profile="combined63",
            device=device,
        )
        model.fit(
            native.iloc[train_idx],
            y[train_idx],
            categorical_feature=[c for c in cats if c in native.columns],
        )
        pred = model.predict_proba(native.iloc[valid_idx])[:, 1]
        oof[valid_idx] = pred
        fold_auc.append(float(roc_auc_score(y[valid_idx], pred)))
    return {
        "oof_auc": float(roc_auc_score(y, oof)),
        "fold_auc": fold_auc,
        "fold_auc_mean": float(np.mean(fold_auc)),
        "fold_auc_std": float(np.std(fold_auc)),
        "feature_count": int(X.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="artifacts/frequency_family_ablation.json")
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--estimators", type=int, default=700)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    if args.rows < args.folds * 20:
        parser.error("--rows is too small for a stable stratified multi-fold screen")
    if args.estimators <= 0:
        parser.error("--estimators must be positive")

    train, test, _ = load_competition(args.data_dir)
    y_full = train[TARGET].astype(int).to_numpy()
    subset = _sample_rows(y_full, min(args.rows, len(train)), args.seed)
    df = train.iloc[subset].reset_index(drop=True)
    y = df[TARGET].astype(int).to_numpy()
    full_reference = (train.drop(columns=[TARGET]), test)
    X, _ = build_features(
        df.drop(columns=[TARGET]),
        test.iloc[:1].copy(),
        use_frequency=True,
        frequency_reference=full_reference,
    )

    groups = frequency_feature_groups(X.columns)
    all_frequency = {column for values in groups.values() for column in values}
    backbone = [column for column in X.columns if column not in all_frequency]
    folds = frozen_folds(y, n_splits=args.folds, seed=args.seed)

    results = {}
    for arm in FREQUENCY_ARMS:
        selected = backbone + frequency_columns_for_arm(X.columns, arm)
        metrics = _evaluate_arm(
            X[selected],
            y,
            folds,
            estimators=args.estimators,
            seed=args.seed,
            device=args.device,
        )
        results[arm] = metrics
        print(
            f"{arm:20s} auc={metrics['oof_auc']:.7f} features={metrics['feature_count']}",
            flush=True,
        )

    full_auc = results["full"]["oof_auc"]
    for arm, metrics in results.items():
        metrics["delta_vs_full"] = float(metrics["oof_auc"] - full_auc)

    payload = {
        "experiment": "frequency_family_ablation",
        "scientific_status": "directional mature-capacity screen; not promotion evidence",
        "rows": int(len(df)),
        "estimators": int(args.estimators),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "device": args.device,
        "frequency_groups": groups,
        "results": results,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
