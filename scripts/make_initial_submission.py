#!/usr/bin/env python
"""Fit the mature dual-view Frontier model on all labeled rows and emit Kaggle submissions.

This script is for submission generation, not model selection. The default 1000-tree
counts and 62.5/37.5 rank blend come from repeated S1 experiments; authoritative S3
iteration counts and weights should replace them once available.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s6e8 import __version__
from s6e8.artifacts import atomic_write_json, runtime_manifest, sha256_file
from s6e8.config import ID_COL, TARGET
from s6e8.features import build_features
from s6e8.io import load_competition
from s6e8.models import make_lgb
from s6e8.preprocess import prepare_tree_frames
from s6e8.submission import rank_blend, unit_rank, write_submission


def fit_predict_lgb(
    X,
    y,
    T,
    cats,
    *,
    seed: int,
    estimators: int,
    profile: str,
    device: str,
    model_path: Path | None,
):
    t0 = time.time()
    model = make_lgb(seed, estimators, profile, device=device)
    model.fit(X, y, categorical_feature=cats)
    pred = model.predict_proba(T)[:, 1]
    elapsed = time.time() - t0
    if not np.isfinite(pred).all():
        raise RuntimeError(f"{profile} produced non-finite test predictions")
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(model_path))
    return pred, elapsed


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train the frozen NOMOPHOBIA v0.1 dual-view model and write validated submissions."
    )
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="artifacts/submissions/initial_v0_1")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--combined-estimators", type=int, default=1000)
    ap.add_argument("--raw-estimators", type=int, default=1000)
    ap.add_argument("--raw-weight", type=float, default=0.375)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument(
        "--hash-inputs",
        action="store_true",
        help="SHA-256 hash the three Kaggle CSVs in metadata.json for byte-level provenance.",
    )
    args = ap.parse_args()
    if args.combined_estimators <= 0 or args.raw_estimators <= 0:
        ap.error("estimator counts must be positive")
    if not 0.0 <= args.raw_weight <= 1.0:
        ap.error("--raw-weight must be in [0, 1]")
    return args


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, test, sample = load_competition(args.data_dir)
    y = train[TARGET].astype(int).to_numpy()
    train_x = train.drop(columns=[TARGET])
    print(
        f"train={len(train):,} test={len(test):,} positive={y.mean():.6f} device={args.device}",
        flush=True,
    )

    t0 = time.time()
    Xc, Tc = build_features(
        train_x,
        test,
        use_frequency=True,
        frequency_reference=(train_x, test),
    )
    feature_seconds = time.time() - t0
    n_features = int(Xc.shape[1])
    print(
        f"combined_features={n_features} feature_seconds={feature_seconds:.1f}",
        flush=True,
    )
    _, _, Xc, Tc, ccats = prepare_tree_frames(Xc, Tc)
    pc, sec_c = fit_predict_lgb(
        Xc,
        y,
        Tc,
        ccats,
        seed=args.seed,
        estimators=args.combined_estimators,
        profile="combined63",
        device=args.device,
        model_path=(out / "models" / "combined63.txt") if args.save_models else None,
    )
    np.save(out / "test_combined_probability.npy", pc)
    print(f"combined_fit_predict_seconds={sec_c:.1f}", flush=True)
    del Xc, Tc
    gc.collect()

    Xr = train_x.drop(columns=[ID_COL], errors="ignore").copy()
    Tr = test.drop(columns=[ID_COL], errors="ignore").copy()
    _, _, Xr, Tr, rcats = prepare_tree_frames(Xr, Tr)
    pr, sec_r = fit_predict_lgb(
        Xr,
        y,
        Tr,
        rcats,
        seed=args.seed + 101,
        estimators=args.raw_estimators,
        profile="raw63",
        device=args.device,
        model_path=(out / "models" / "raw63.txt") if args.save_models else None,
    )
    np.save(out / "test_raw_probability.npy", pr)
    print(f"raw_fit_predict_seconds={sec_r:.1f}", flush=True)

    predictions = {"combined": pc, "raw": pr}
    ranked = {name: unit_rank(pred) for name, pred in predictions.items()}
    candidate_weights = sorted({0.0, 0.325, args.raw_weight, 0.375, 0.425})
    portfolio = {}
    for raw_weight in candidate_weights:
        score = rank_blend(
            predictions,
            {"combined": 1.0 - raw_weight, "raw": raw_weight},
        )
        label = (
            f"dualview_raw{int(round(raw_weight * 1000)):03d}"
            if raw_weight > 0
            else "combined_only"
        )
        portfolio[label] = write_submission(
            out / f"submission_{label}.csv", sample, test, score
        )

    primary_score = rank_blend(
        predictions,
        {"combined": 1.0 - args.raw_weight, "raw": args.raw_weight},
    )
    primary_artifact = write_submission(
        out / "submission.csv", sample, test, primary_score
    )

    metadata = {
        "version": f"nomophobia-{__version__}",
        "scientific_status": (
            "initial submission candidate; S1-derived configuration, not S3-promoted"
        ),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "seed": int(args.seed),
        "device": args.device,
        "combined_estimators": int(args.combined_estimators),
        "raw_estimators": int(args.raw_estimators),
        "primary_raw_weight": float(args.raw_weight),
        "feature_count_combined": n_features,
        "feature_seconds": float(feature_seconds),
        "fit_seconds": {"combined": float(sec_c), "raw": float(sec_r)},
        "prediction_correlation_probability": float(np.corrcoef(pc, pr)[0, 1]),
        "prediction_correlation_rank": float(
            np.corrcoef(ranked["combined"], ranked["raw"])[0, 1]
        ),
        "primary": primary_artifact,
        "portfolio": portfolio,
        "runtime": runtime_manifest(),
    }
    if args.hash_inputs:
        data_dir = Path(args.data_dir)
        metadata["input_sha256"] = {
            name: sha256_file(data_dir / name)
            for name in ("train.csv", "test.csv", "sample_submission.csv")
        }

    atomic_write_json(out / "metadata.json", metadata)
    metadata["metadata_sha256"] = sha256_file(out / "metadata.json")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
