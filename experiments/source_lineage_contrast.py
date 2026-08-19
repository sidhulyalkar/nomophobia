#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frontier_contrast_campaign import (
    _fit_predict,
    _load_anchor,
    _load_control_artifacts,
    _parse_weights,
    _read_competition,
)
from s6e8.config import ID_COL, RAW_COLS, TARGET
from s6e8.contrast import (
    ResidualGate,
    apply_rank_residual,
    rank_direction,
    rotating_residual_gate,
)
from s6e8.identity import build_contrast_feature_frame
from s6e8.source_lineage import (
    add_source_lineage_features,
    encoder_report,
    fit_source_lineage_encoder,
)
from s6e8.submission import write_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    train: pd.DataFrame,
    lineage_reliability: np.ndarray,
) -> dict:
    reliability_rank = (
        pd.Series(lineage_reliability)
        .rank(method="average", pct=True)
        .to_numpy(float)
    )
    anchor_rank = (
        pd.Series(anchor).rank(method="average", pct=True).to_numpy(float)
    )
    missing = train[RAW_COLS].isna().sum(axis=1).to_numpy(int)

    masks = {
        "lineage_low": reliability_rank <= 0.25,
        "lineage_mid": (reliability_rank > 0.25) & (reliability_rank <= 0.75),
        "lineage_high": reliability_rank > 0.75,
        "anchor_low": anchor_rank <= 0.20,
        "anchor_boundary": (anchor_rank > 0.35) & (anchor_rank <= 0.65),
        "anchor_high": anchor_rank > 0.80,
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
            "Test exact original-source subset lineage as a matched fixed-schedule "
            "residual against an aligned frontier anchor."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--anchor-oof", required=True)
    parser.add_argument("--anchor-test", required=True)
    parser.add_argument("--anchor-oof-col", default="honest_blend")
    parser.add_argument("--anchor-test-col", default=TARGET)
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument(
        "--mode", choices=["membership", "lineage"], default="membership"
    )
    parser.add_argument("--rounds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=96)
    parser.add_argument("--screen-rows", type=int, default=60_000)
    parser.add_argument("--source-smoothing", type=float, default=8.0)
    parser.add_argument("--reuse-control-dir", default=None)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--min-fold-wins", type=int, default=4)
    parser.add_argument("--fold-tolerance", type=float, default=-2e-6)
    parser.add_argument("--slice-tolerance", type=float, default=-2e-6)
    parser.add_argument("--structural-tolerance", type=float, default=-5e-5)
    parser.add_argument("--out-dir", default="artifacts/source_lineage")
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    source_path = Path(args.source_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = _parse_weights(args.weights)

    train, test, sample = _read_competition(data_dir)
    # The source uses the literal string "None" as an ordinal severity class.
    # Pandas' default NA vocabulary can consume that token, so preserve strings.
    source = pd.read_csv(source_path, keep_default_na=False)
    y = train[TARGET].to_numpy(np.int8)
    ids = train[ID_COL].to_numpy()
    anchor_oof, anchor_test, folds = _load_anchor(
        Path(args.anchor_oof),
        Path(args.anchor_test),
        train,
        test,
        oof_col=args.anchor_oof_col,
        test_col=args.anchor_test_col,
        fold_col=args.fold_col,
    )
    unique_folds = np.unique(folds)
    if len(unique_folds) < 3:
        raise ValueError("at least three aligned outer folds are required")

    reference = pd.concat(
        [train[RAW_COLS], test[RAW_COLS]], ignore_index=True
    )
    encoder = fit_source_lineage_encoder(
        source,
        reference,
        max_order=args.max_order,
        max_groups=args.max_groups,
        screen_rows=args.screen_rows,
        seed=args.seed,
    )
    include_source_labels = args.mode == "lineage"
    treatment_full = add_source_lineage_features(
        train,
        encoder,
        include_source_labels=include_source_labels,
        source_smoothing=args.source_smoothing,
    )
    treatment_test = add_source_lineage_features(
        test,
        encoder,
        include_source_labels=include_source_labels,
        source_smoothing=args.source_smoothing,
    )
    control_full = build_contrast_feature_frame(train, "raw")
    control_test = build_contrast_feature_frame(test, "raw")
    reliability = treatment_full["srcagg__seen_fraction"].to_numpy(float)

    if args.reuse_control_dir:
        oof_control, test_control = _load_control_artifacts(
            Path(args.reuse_control_dir),
            n_train=len(train),
            n_test=len(test),
            family="lgb",
            rounds=args.rounds,
            folds=folds,
        )
        train_control = False
    else:
        oof_control = np.empty(len(train), dtype=float)
        test_control = np.zeros(len(test), dtype=float)
        train_control = True

    oof_treatment = np.empty(len(train), dtype=float)
    test_treatment = np.zeros(len(test), dtype=float)
    fold_metrics = []

    for fold in unique_folds:
        fold_started = time.time()
        train_idx = np.where(folds != fold)[0]
        valid_idx = np.where(folds == fold)[0]
        yy = y[train_idx]
        yv = y[valid_idx]

        if train_control:
            pred_control, test_part_control = _fit_predict(
                "lgb",
                control_full.iloc[train_idx].reset_index(drop=True),
                yy,
                control_full.iloc[valid_idx].reset_index(drop=True),
                control_test.reset_index(drop=True),
                seed=args.seed + int(fold),
                rounds=args.rounds,
                device=args.device,
            )
            oof_control[valid_idx] = pred_control
            test_control += test_part_control / len(unique_folds)
        else:
            pred_control = oof_control[valid_idx]

        pred_treatment, test_part_treatment = _fit_predict(
            "lgb",
            treatment_full.iloc[train_idx].reset_index(drop=True),
            yy,
            treatment_full.iloc[valid_idx].reset_index(drop=True),
            treatment_test.reset_index(drop=True),
            seed=args.seed + int(fold),
            rounds=args.rounds,
            device=args.device,
        )
        oof_treatment[valid_idx] = pred_treatment
        test_treatment += test_part_treatment / len(unique_folds)

        control_auc = float(roc_auc_score(yv, pred_control))
        treatment_auc = float(roc_auc_score(yv, pred_treatment))
        row = {
            "fold": int(fold),
            "control_auc": control_auc,
            "treatment_auc": treatment_auc,
            "treatment_minus_control": treatment_auc - control_auc,
            "rank_corr": float(
                pd.Series(pred_control).rank(method="average").corr(
                    pd.Series(pred_treatment).rank(method="average")
                )
            ),
            "seconds": round(time.time() - fold_started, 2),
        }
        fold_metrics.append(row)
        print(json.dumps(row), flush=True)

    direction_oof = rank_direction(oof_treatment, oof_control)
    direction_test = rank_direction(test_treatment, test_control)
    gate = ResidualGate(
        min_gain=args.min_gain,
        min_fold_wins=args.min_fold_wins,
        fold_tolerance=args.fold_tolerance,
        require_id_slices=True,
        slice_tolerance=args.slice_tolerance,
    )
    decision = rotating_residual_gate(
        y, anchor_oof, direction_oof, folds, weights, ids=ids, gate=gate
    )
    honest = decision.pop("honest_oof")
    structural = _structural_stability(
        y, anchor_oof, honest, train, reliability
    )
    structural_pass = bool(
        structural["worst_delta"] is not None
        and structural["worst_delta"] >= args.structural_tolerance
    )
    accepted = bool(decision["accepted"] and structural_pass)
    deploy_weight = float(decision["deploy_weight"]) if accepted else 0.0
    diagnostic_weight = float(decision["selected_weight_median"])
    diagnostic_oof = apply_rank_residual(
        anchor_oof, direction_oof, diagnostic_weight
    )
    diagnostic_test = apply_rank_residual(
        anchor_test, direction_test, diagnostic_weight
    )
    gated_test = (
        apply_rank_residual(anchor_test, direction_test, deploy_weight)
        if accepted
        else anchor_test.copy()
    )

    np.save(out / "folds.npy", folds)
    np.save(out / "oof_control.npy", oof_control)
    np.save(out / "oof_treatment.npy", oof_treatment)
    np.save(out / "test_control.npy", test_control)
    np.save(out / "test_treatment.npy", test_treatment)
    np.save(out / "direction_oof.npy", direction_oof)
    np.save(out / "direction_test.npy", direction_test)
    np.save(out / "oof_honest_residual.npy", honest)

    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "fold": folds,
            "anchor": anchor_oof,
            "control": oof_control,
            "treatment": oof_treatment,
            "direction": direction_oof,
            "lineage_reliability": reliability,
            "honest_residual": honest,
            "diagnostic_candidate": diagnostic_oof,
        }
    ).to_csv(out / "oof.csv", index=False)

    diagnostic_stats = write_submission(
        out / "submission_diagnostic.csv",
        sample,
        test,
        diagnostic_test,
    )
    gated_stats = write_submission(
        out / "submission_gated.csv", sample, test, gated_test
    )
    screen_report = encoder_report(encoder)
    (out / "source_lineage_screen.json").write_text(
        json.dumps(screen_report, indent=2) + "\n"
    )

    report = {
        "version": "source-lineage-contrast-v1",
        "mode": args.mode,
        "hypothesis": (
            "exact partial source-row membership retains generator lineage that is "
            "not represented by nearest-source distance or a smooth source prior"
        ),
        "source": {
            "file": str(source_path),
            "sha256": _sha256(source_path),
            "rows": int(len(source)),
            "uses_source_labels": include_source_labels,
            "smoothing": args.source_smoothing if include_source_labels else None,
        },
        "screen": screen_report,
        "rounds": args.rounds,
        "seed": args.seed,
        "device": args.device,
        "weights": weights,
        "features": {
            "control": int(control_full.shape[1]),
            "treatment": int(treatment_full.shape[1]),
        },
        "standalone": {
            "anchor_auc": float(roc_auc_score(y, anchor_oof)),
            "control_auc": float(roc_auc_score(y, oof_control)),
            "treatment_auc": float(roc_auc_score(y, oof_treatment)),
            "treatment_minus_control": float(
                roc_auc_score(y, oof_treatment)
                - roc_auc_score(y, oof_control)
            ),
            "control_treatment_rank_corr": float(
                pd.Series(oof_control).rank(method="average").corr(
                    pd.Series(oof_treatment).rank(method="average")
                )
            ),
        },
        "fold_metrics": fold_metrics,
        "residual_gate": decision,
        "structural_stability": structural,
        "structural_tolerance": args.structural_tolerance,
        "accepted": accepted,
        "deploy_weight": deploy_weight,
        "diagnostic_submission": diagnostic_stats,
        "gated_submission": gated_stats,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()