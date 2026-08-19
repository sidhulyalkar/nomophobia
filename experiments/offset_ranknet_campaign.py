#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frontier_contrast_campaign import _load_anchor, _parse_weights, _read_competition
from s6e8.config import ID_COL, TARGET
from s6e8.contrast import ResidualGate, apply_rank_residual, rank01, rank_direction, rotating_residual_gate
from s6e8.offset_ranknet import fit_offset_ranknet, offset_ranknet_report, score_offset_ranknet
from s6e8.submission import write_submission


def _load_v19(path: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    if ID_COL not in frame or TARGET not in frame:
        raise ValueError(f"{path} must contain {ID_COL} and {TARGET}")
    if len(frame) != len(ids) or not np.array_equal(frame[ID_COL].to_numpy(), ids):
        raise ValueError(f"{path} IDs do not align")
    values = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fit a champion-offset pairwise RankNet residual.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--v19-oof", required=True)
    parser.add_argument("--v19-test", required=True)
    parser.add_argument("--lgb-direction-dir", required=True)
    parser.add_argument("--lgb-weight", type=float, default=0.015)
    parser.add_argument("--expected-anchor-auc", type=float, default=0.9701014001897124)
    parser.add_argument("--max-pairs", type=int, default=240000)
    parser.add_argument("--uniform-fraction", type=float, default=0.50)
    parser.add_argument("--anchor-temperature", type=float, default=12.0)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--optimizer-iterations", type=int, default=28)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--weights", default="0,0.05,0.10,0.20,0.35,0.50,0.75,1.0")
    parser.add_argument("--out-dir", default="artifacts/offset_ranknet")
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = _parse_weights(args.weights)
    train, test, sample = _read_competition(Path(args.data_dir))
    y = train[TARGET].to_numpy(np.int8)
    ids = train[ID_COL].to_numpy()
    test_ids = test[ID_COL].to_numpy()
    v19_oof = _load_v19(Path(args.v19_oof), ids)
    v19_test = _load_v19(Path(args.v19_test), test_ids)

    lgb_dir = Path(args.lgb_direction_dir)
    direction_oof = np.load(lgb_dir / "direction_oof.npy")
    direction_test = np.load(lgb_dir / "direction_test.npy")
    folds = np.load(lgb_dir / "folds.npy")
    champion_oof = apply_rank_residual(v19_oof, direction_oof, args.lgb_weight)
    champion_test = apply_rank_residual(v19_test, direction_test, args.lgb_weight)
    champion_auc = float(roc_auc_score(y, champion_oof))
    if abs(champion_auc - args.expected_anchor_auc) > 5e-10:
        raise ValueError(f"mature anchor mismatch: {champion_auc:.12f}")

    unique_folds = np.unique(folds)
    candidate_oof = np.empty(len(train), dtype=float)
    candidate_test = np.zeros(len(test), dtype=float)
    fold_reports = []
    offsets = (64, 256, 1024, 4096, 16384, 65536, 131072)
    for fold in unique_folds:
        fold_started = time.time()
        train_idx = np.where(folds != fold)[0]
        valid_idx = np.where(folds == fold)[0]
        model = fit_offset_ranknet(
            train.iloc[train_idx],
            y[train_idx],
            champion_oof[train_idx],
            max_pairs=args.max_pairs,
            uniform_fraction=args.uniform_fraction,
            offsets=offsets,
            anchor_temperature=args.anchor_temperature,
            l2=args.l2,
            max_iter=args.optimizer_iterations,
            seed=args.seed + int(fold),
        )
        valid_score = score_offset_ranknet(model, train.iloc[valid_idx], champion_oof[valid_idx])
        test_score = score_offset_ranknet(model, test, champion_test)
        candidate_oof[valid_idx] = rank01(valid_score)
        candidate_test += rank01(test_score) / len(unique_folds)
        row = {
            "fold": int(fold),
            "anchor_auc": float(roc_auc_score(y[valid_idx], champion_oof[valid_idx])),
            "candidate_auc": float(roc_auc_score(y[valid_idx], valid_score)),
            "delta": float(
                roc_auc_score(y[valid_idx], valid_score)
                - roc_auc_score(y[valid_idx], champion_oof[valid_idx])
            ),
            "rank_corr": float(
                pd.Series(valid_score).rank(method="average").corr(
                    pd.Series(champion_oof[valid_idx]).rank(method="average")
                )
            ),
            "model": offset_ranknet_report(model),
            "seconds": round(time.time() - fold_started, 2),
        }
        fold_reports.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "model"}), flush=True)

    residual_oof = rank_direction(candidate_oof, champion_oof)
    residual_test = rank_direction(candidate_test, champion_test)
    gate = ResidualGate(
        min_gain=1e-6,
        min_fold_wins=4,
        fold_tolerance=-2e-6,
        require_id_slices=True,
        slice_tolerance=-2e-6,
    )
    decision = rotating_residual_gate(
        y,
        champion_oof,
        residual_oof,
        folds,
        weights,
        ids=ids,
        gate=gate,
    )
    honest = decision.pop("honest_oof")
    accepted = bool(decision["accepted"])
    deploy_weight = float(decision["deploy_weight"]) if accepted else 0.0
    deploy_test = (
        apply_rank_residual(champion_test, residual_test, deploy_weight)
        if accepted
        else champion_test.copy()
    )
    direct_oof_auc = float(roc_auc_score(y, candidate_oof))
    submission_stats = write_submission(out / "submission_promoted.csv", sample, test, deploy_test)
    np.save(out / "folds.npy", folds)
    np.save(out / "candidate_oof.npy", candidate_oof)
    np.save(out / "candidate_test.npy", candidate_test)
    np.save(out / "direction_oof.npy", residual_oof)
    np.save(out / "direction_test.npy", residual_test)
    pd.DataFrame(
        {
            ID_COL: ids,
            TARGET: y,
            "fold": folds,
            "champion": champion_oof,
            "offset_ranknet": candidate_oof,
            "honest_candidate": honest,
        }
    ).to_csv(out / "oof.csv", index=False)
    report = {
        "version": "champion-offset-ranknet-v1",
        "anchor_auc": champion_auc,
        "direct_candidate_auc": direct_oof_auc,
        "direct_gain": direct_oof_auc - champion_auc,
        "fold_reports": fold_reports,
        "weights": weights,
        "residual_gate": decision,
        "accepted": accepted,
        "deploy_weight": deploy_weight,
        "submission": submission_stats,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
