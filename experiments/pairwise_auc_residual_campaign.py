#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frontier_contrast_campaign import _parse_weights, _read_competition
from s6e8.config import ID_COL, RAW_COLS, TARGET
from s6e8.contrast import (
    ResidualGate,
    apply_rank_residual,
    orthogonalize_train_test,
    rank01,
    rotating_residual_gate,
)
from s6e8.pairwise_auc import (
    fit_pairwise_ranker,
    pairwise_ranker_report,
    score_pairwise_ranker,
)
from s6e8.submission import write_submission


def _load_vector_csv(path: Path, ids: np.ndarray, column: str) -> np.ndarray:
    frame = pd.read_csv(path)
    if ID_COL not in frame or column not in frame:
        raise ValueError(f"{path} must contain {ID_COL!r} and {column!r}")
    if len(frame) != len(ids) or not np.array_equal(frame[ID_COL].to_numpy(), ids):
        raise ValueError(f"{path} IDs do not align")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return values


def _safe_delta(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    if int(mask.sum()) < 200 or len(np.unique(y[mask])) < 2:
        return None
    return float(
        roc_auc_score(y[mask], candidate[mask])
        - roc_auc_score(y[mask], anchor[mask])
    )


def _structural_stability(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    correction: np.ndarray,
    direction: np.ndarray,
    train: pd.DataFrame,
) -> dict:
    anchor_rank = rank01(anchor)
    correction_strength = rank01(np.abs(correction - np.median(correction)))
    direction_strength = rank01(np.abs(direction))
    missing = train[RAW_COLS].isna().sum(axis=1).to_numpy(int)
    masks = {
        "anchor_low": anchor_rank <= 0.20,
        "anchor_boundary": (anchor_rank > 0.35) & (anchor_rank <= 0.65),
        "anchor_high": anchor_rank > 0.80,
        "correction_high": correction_strength > 0.75,
        "direction_high": direction_strength > 0.75,
        "missing_0": missing == 0,
        "missing_1": missing == 1,
        "missing_2plus": missing >= 2,
    }
    deltas = {
        name: _safe_delta(y, anchor, candidate, mask)
        for name, mask in masks.items()
    }
    finite = [value for value in deltas.values() if value is not None]
    return {
        "deltas": deltas,
        "worst_delta": None if not finite else float(min(finite)),
        "wins": int(sum(value > 0 for value in finite)),
        "total": len(finite),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit a pairwise AUC ranker on local errors of the mature v19+LGB "
            "champion, then admit only a held-fold-stable orthogonal correction."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--v19-oof", required=True)
    parser.add_argument("--v19-test", required=True)
    parser.add_argument("--lgb-direction-dir", required=True)
    parser.add_argument("--lgb-weight", type=float, default=0.015)
    parser.add_argument("--expected-anchor-auc", type=float, default=0.9701014001897124)
    parser.add_argument("--anchor-tolerance", type=float, default=5e-10)
    parser.add_argument("--max-pairs", type=int, default=210000)
    parser.add_argument("--ranker-alpha", type=float, default=1e-4)
    parser.add_argument("--ranker-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--min-fold-wins", type=int, default=4)
    parser.add_argument("--fold-tolerance", type=float, default=-2e-6)
    parser.add_argument("--slice-tolerance", type=float, default=-2e-6)
    parser.add_argument("--structural-tolerance", type=float, default=-5e-5)
    parser.add_argument("--out-dir", default="artifacts/pairwise_auc")
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = _parse_weights(args.weights)

    train, test, sample = _read_competition(Path(args.data_dir))
    y = train[TARGET].to_numpy(np.int8)
    ids = train[ID_COL].to_numpy()
    test_ids = test[ID_COL].to_numpy()
    v19_oof = _load_vector_csv(Path(args.v19_oof), ids, TARGET)
    v19_test = _load_vector_csv(Path(args.v19_test), test_ids, TARGET)

    direction_dir = Path(args.lgb_direction_dir)
    lgb_direction_oof = np.load(direction_dir / "direction_oof.npy")
    lgb_direction_test = np.load(direction_dir / "direction_test.npy")
    folds = np.load(direction_dir / "folds.npy")
    if not (
        len(lgb_direction_oof) == len(train)
        and len(lgb_direction_test) == len(test)
        and len(folds) == len(train)
    ):
        raise ValueError("LGB direction artifact dimensions do not align")

    champion_oof = apply_rank_residual(v19_oof, lgb_direction_oof, args.lgb_weight)
    champion_test = apply_rank_residual(v19_test, lgb_direction_test, args.lgb_weight)
    champion_auc = float(roc_auc_score(y, champion_oof))
    if abs(champion_auc - args.expected_anchor_auc) > args.anchor_tolerance:
        raise ValueError(
            f"reconstructed mature anchor AUC {champion_auc:.12f} does not match "
            f"expected {args.expected_anchor_auc:.12f}"
        )

    unique_folds = np.unique(folds)
    correction_oof = np.empty(len(train), dtype=float)
    correction_test = np.zeros(len(test), dtype=float)
    fold_reports = []

    for fold in unique_folds:
        fold_started = time.time()
        train_idx = np.where(folds != fold)[0]
        valid_idx = np.where(folds == fold)[0]
        model = fit_pairwise_ranker(
            train.iloc[train_idx],
            y[train_idx],
            champion_oof[train_idx],
            max_pairs=args.max_pairs,
            alpha=args.ranker_alpha,
            max_iter=args.ranker_iterations,
            seed=args.seed + int(fold),
        )
        valid_score = score_pairwise_ranker(model, train.iloc[valid_idx])
        test_score = score_pairwise_ranker(model, test)
        correction_oof[valid_idx] = rank01(valid_score)
        correction_test += rank01(test_score) / len(unique_folds)
        row = {
            "fold": int(fold),
            "correction_auc": float(roc_auc_score(y[valid_idx], valid_score)),
            "anchor_auc": float(roc_auc_score(y[valid_idx], champion_oof[valid_idx])),
            "rank_corr_with_anchor": float(
                pd.Series(valid_score).rank(method="average").corr(
                    pd.Series(champion_oof[valid_idx]).rank(method="average")
                )
            ),
            "ranker": pairwise_ranker_report(model),
            "seconds": round(time.time() - fold_started, 2),
        }
        fold_reports.append(row)
        print(
            json.dumps({key: value for key, value in row.items() if key != "ranker"}),
            flush=True,
        )

    raw_oof = rank01(correction_oof) - 0.5
    raw_test = rank01(correction_test) - 0.5
    orth_oof, orth_test = orthogonalize_train_test(
        raw_oof,
        raw_test,
        [rank01(champion_oof), lgb_direction_oof],
        [rank01(champion_test), lgb_direction_test],
    )
    scale = float(np.quantile(np.abs(orth_oof), 0.99))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("pairwise correction collapsed after orthogonalization")
    direction_oof = np.clip(orth_oof / scale, -1.0, 1.0)
    direction_test = np.clip(orth_test / scale, -1.0, 1.0)

    gate = ResidualGate(
        min_gain=args.min_gain,
        min_fold_wins=args.min_fold_wins,
        fold_tolerance=args.fold_tolerance,
        require_id_slices=True,
        slice_tolerance=args.slice_tolerance,
    )
    decision = rotating_residual_gate(
        y,
        champion_oof,
        direction_oof,
        folds,
        weights,
        ids=ids,
        gate=gate,
    )
    honest = decision.pop("honest_oof")
    structural = _structural_stability(
        y,
        champion_oof,
        honest,
        correction_oof,
        direction_oof,
        train,
    )
    structural_pass = bool(
        structural["worst_delta"] is not None
        and structural["worst_delta"] >= args.structural_tolerance
    )
    accepted = bool(decision["accepted"] and structural_pass)
    deploy_weight = float(decision["deploy_weight"]) if accepted else 0.0
    diagnostic_weight = float(decision["selected_weight_median"])
    diagnostic_test = apply_rank_residual(
        champion_test, direction_test, diagnostic_weight
    )
    promoted_test = (
        apply_rank_residual(champion_test, direction_test, deploy_weight)
        if accepted
        else champion_test.copy()
    )

    np.save(out / "folds.npy", folds)
    np.save(out / "correction_oof.npy", correction_oof)
    np.save(out / "correction_test.npy", correction_test)
    np.save(out / "direction_oof.npy", direction_oof)
    np.save(out / "direction_test.npy", direction_test)
    np.save(out / "oof_honest_residual.npy", honest)
    pd.DataFrame(
        {
            ID_COL: ids,
            TARGET: y,
            "fold": folds,
            "v19": v19_oof,
            "champion": champion_oof,
            "pairwise_correction": correction_oof,
            "direction": direction_oof,
            "honest_candidate": honest,
        }
    ).to_csv(out / "oof.csv", index=False)

    diagnostic_stats = write_submission(
        out / "submission_diagnostic.csv", sample, test, diagnostic_test
    )
    promoted_stats = write_submission(
        out / "submission_promoted.csv", sample, test, promoted_test
    )
    report = {
        "version": "pairwise-auc-residual-v1",
        "hypothesis": (
            "a local pairwise ranker trained only on opposite-class rows that the "
            "mature champion nearly ties can learn systematic AUC ordering errors"
        ),
        "anchor": {
            "v19_auc": float(roc_auc_score(y, v19_oof)),
            "lgb_weight": args.lgb_weight,
            "champion_auc": champion_auc,
        },
        "ranker": {
            "max_pairs": args.max_pairs,
            "alpha": args.ranker_alpha,
            "iterations": args.ranker_iterations,
            "seed": args.seed,
            "fold_reports": fold_reports,
            "global_correction_auc": float(roc_auc_score(y, correction_oof)),
            "anchor_correction_rank_corr": float(
                pd.Series(champion_oof).rank(method="average").corr(
                    pd.Series(correction_oof).rank(method="average")
                )
            ),
            "orthogonal_scale_q99": scale,
            "orthogonal_anchor_corr": float(
                np.corrcoef(direction_oof, rank01(champion_oof))[0, 1]
            ),
            "orthogonal_lgb_corr": float(
                np.corrcoef(direction_oof, lgb_direction_oof)[0, 1]
            ),
        },
        "weights": weights,
        "residual_gate": decision,
        "structural_stability": structural,
        "structural_tolerance": args.structural_tolerance,
        "accepted": accepted,
        "deploy_weight": deploy_weight,
        "diagnostic_submission": diagnostic_stats,
        "promoted_submission": promoted_stats,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
