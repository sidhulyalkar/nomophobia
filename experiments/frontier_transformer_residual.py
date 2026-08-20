#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from s6e8.config import ID_COL, RAW_COLS, TARGET
from s6e8.contrast import ResidualGate, apply_rank_residual, rank01, rotating_residual_gate
from s6e8.submission import write_submission
from s6e8.transformer import TokenTransformer, _fit_epochs, _matrix, _predict
from s6e8.validation import validate_competition_frames

DEFAULT_WEIGHTS = [0.0, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.075, 0.10]


def _load_anchor(oof_path: Path, test_path: Path, train: pd.DataFrame, test: pd.DataFrame):
    oof = pd.read_csv(oof_path)
    pred = pd.read_csv(test_path)
    for c in (ID_COL, TARGET, "fold", "honest_blend"):
        if c not in oof:
            raise ValueError(f"anchor OOF missing {c}")
    if not np.array_equal(oof[ID_COL].to_numpy(), train[ID_COL].to_numpy()):
        raise ValueError("anchor OOF IDs are not aligned")
    if not np.array_equal(oof[TARGET].to_numpy(np.int8), train[TARGET].to_numpy(np.int8)):
        raise ValueError("anchor OOF target mismatch")
    if not np.array_equal(pred[ID_COL].to_numpy(), test[ID_COL].to_numpy()):
        raise ValueError("anchor test IDs are not aligned")
    return oof["honest_blend"].to_numpy(float), pred[TARGET].to_numpy(float), oof["fold"].to_numpy(int)


def _parse_weights(text: str | None):
    if not text:
        return DEFAULT_WEIGHTS.copy()
    vals = [float(v) for v in text.split(",") if v.strip()]
    if not vals or vals[0] != 0.0:
        vals = [0.0, *vals]
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError("weights must be strictly increasing")
    return vals


def main():
    ap = argparse.ArgumentParser(description="Fixed-schedule neural residual against the aligned quad anchor")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--anchor-oof", required=True)
    ap.add_argument("--anchor-test", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out-dir", default="artifacts/transformer_residual")
    a = ap.parse_args()

    started = time.time()
    data = Path(a.data_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    validate_competition_frames(train, test, sample)
    y = train[TARGET].to_numpy(np.float32)
    ids = train[ID_COL].to_numpy()
    anchor_oof, anchor_test, folds = _load_anchor(Path(a.anchor_oof), Path(a.anchor_test), train, test)

    X, M, T, TM = _matrix(train[RAW_COLS], test[RAW_COLS])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.empty(len(train), dtype=float)
    test_parts = []
    fold_metrics = []
    lossfn = torch.nn.BCEWithLogitsLoss()

    for f in sorted(np.unique(folds)):
        ti = np.where(folds != f)[0]
        vi = np.where(folds == f)[0]
        torch.manual_seed(a.seed + 1000 + int(f))
        np.random.seed(a.seed + 1000 + int(f))
        model = TokenTransformer(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        t0 = time.time()
        _fit_epochs(model, opt, lossfn, X, M, y, ti, a.epochs, a.batch_size, device)
        pv = _predict(model, X, M, vi, a.batch_size, device)
        pt = _predict(model, T, TM, np.arange(len(T)), a.batch_size, device)
        oof[vi] = pv
        test_parts.append(pt)
        row = {"fold": int(f), "auc": float(roc_auc_score(y[vi], pv)), "epochs": int(a.epochs), "seconds": round(time.time() - t0, 2)}
        fold_metrics.append(row)
        print(json.dumps(row), flush=True)

    test_pred = np.mean(test_parts, axis=0)
    neural_auc = float(roc_auc_score(y, oof))
    direction_oof = rank01(oof) - rank01(anchor_oof)
    direction_test = rank01(test_pred) - rank01(anchor_test)
    gate = ResidualGate(min_gain=1e-6, min_fold_wins=4, fold_tolerance=-2e-6, slice_tolerance=-2e-6)
    decision = rotating_residual_gate(y, anchor_oof, direction_oof, folds, _parse_weights(a.weights), ids=ids, gate=gate)
    deploy_weight = float(decision["deploy_weight"])
    candidate_oof = decision["honest_oof"] if decision["accepted"] else rank01(anchor_oof)
    candidate_test = apply_rank_residual(anchor_test, direction_test, deploy_weight) if deploy_weight > 0 else rank01(anchor_test)

    np.save(out / "oof_transformer.npy", oof)
    np.save(out / "test_transformer.npy", test_pred)
    np.save(out / "direction_oof.npy", direction_oof)
    np.save(out / "direction_test.npy", direction_test)
    np.save(out / "candidate_oof.npy", candidate_oof)
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: train[TARGET], "fold": folds, "prediction": oof}).to_csv(out / "oof.csv", index=False)
    submission = write_submission(out / "submission_candidate.csv", sample, test, candidate_test)
    report = {
        "version": "fixed-transformer-residual-v1",
        "device": str(device),
        "epochs": int(a.epochs),
        "batch_size": int(a.batch_size),
        "anchor_auc": float(roc_auc_score(y, anchor_oof)),
        "transformer_auc": neural_auc,
        "fold_metrics": fold_metrics,
        "accepted": bool(decision["accepted"]),
        "deploy_weight": deploy_weight,
        "selected_weights": decision["selected_weights"],
        "honest_metrics": decision["honest_metrics"],
        "submission": submission,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
