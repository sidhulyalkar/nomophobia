from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _weight_rows(blend: dict[str, Any]) -> tuple[list[str], list[list[float]]]:
    names = list(blend.get("names") or [])
    rotations = blend.get("rotation_weights") or []
    if rotations and not names:
        weights = blend.get("weights") or {}
        names = list(weights)
    return names, [list(map(float, row)) for row in rotations]


def summarize_s3_runs(
    runs: list[str | Path],
    *,
    combined_key: str = "lgb_combined63",
    raw_key: str = "lgb_raw63",
    seed_spread_stop: float = 0.002,
    correlation_stop: float = 0.988,
    weight_std_warn: float = 0.12,
    weight_range_warn: float = 0.35,
    selection_optimism_warn: float = 0.0005,
) -> dict[str, Any]:
    """Aggregate diagnostics that answer whether the dual-view S1 thesis survived S3."""

    seed_rows: list[dict[str, Any]] = []
    all_rotation_weights: dict[str, list[float]] = {}
    correlations: list[float] = []
    blend_aucs: list[float] = []
    optimisms: list[float] = []

    for run_value in runs:
        run = Path(run_value)
        summary = _read_json(run / "run_summary.json")
        blend = _read_json(run / "blend.json")
        fold_seed = summary.get("fold_seed")
        blend_auc = summary.get("blend_auc_honest")
        if blend_auc is not None:
            blend_aucs.append(float(blend_auc))
        optimism = summary.get("blend_selection_optimism")
        if optimism is not None:
            optimisms.append(float(optimism))

        corr_value = None
        corr_path = run / "expert_rank_correlation.csv"
        if corr_path.exists():
            corr = pd.read_csv(corr_path, index_col=0)
            if combined_key in corr.index and raw_key in corr.columns:
                corr_value = float(corr.loc[combined_key, raw_key])
                correlations.append(corr_value)

        names, rotation_rows = _weight_rows(blend)
        seed_weights: dict[str, list[float]] = {name: [] for name in names}
        for row in rotation_rows:
            if len(row) != len(names):
                continue
            for name, value in zip(names, row):
                seed_weights[name].append(float(value))
                all_rotation_weights.setdefault(name, []).append(float(value))

        seed_rows.append(
            {
                "run": str(run),
                "fold_seed": fold_seed,
                "blend_auc_honest": blend_auc,
                "blend_selection_auc": summary.get("blend_selection_auc"),
                "blend_selection_optimism": optimism,
                "rank_correlation_combined_raw": corr_value,
                "rotation_weight_mean": {
                    name: float(np.mean(values)) if values else None
                    for name, values in seed_weights.items()
                },
                "rotation_weight_std": {
                    name: float(np.std(values)) if values else None
                    for name, values in seed_weights.items()
                },
            }
        )

    weight_stats: dict[str, dict[str, float | int]] = {}
    for name, values in all_rotation_weights.items():
        arr = np.asarray(values, dtype=float)
        if not len(arr):
            continue
        weight_stats[name] = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "range": float(arr.max() - arr.min()),
        }

    seed_spread = float(max(blend_aucs) - min(blend_aucs)) if len(blend_aucs) >= 2 else None
    mean_corr = float(np.mean(correlations)) if correlations else None
    max_corr = float(np.max(correlations)) if correlations else None
    mean_optimism = float(np.mean(optimisms)) if optimisms else None

    stop_flags: list[str] = []
    warning_flags: list[str] = []
    if seed_spread is not None and seed_spread > seed_spread_stop:
        stop_flags.append("STOP_SEED_BLEND_AUC_SPREAD")
    if max_corr is not None and max_corr > correlation_stop:
        stop_flags.append("STOP_DUAL_VIEW_RANK_CORRELATION")
    if mean_optimism is not None and mean_optimism > selection_optimism_warn:
        warning_flags.append("WARN_BLEND_SELECTION_OPTIMISM")

    for name, stats in weight_stats.items():
        if stats["std"] > weight_std_warn:
            warning_flags.append(f"WARN_WEIGHT_STD__{name}")
        if stats["range"] > weight_range_warn:
            warning_flags.append(f"WARN_WEIGHT_RANGE__{name}")

    return {
        "version": "nomophobia-v0.3",
        "n_seed_runs": int(len(runs)),
        "seed_runs": seed_rows,
        "blend_auc_by_seed": blend_aucs,
        "blend_auc_seed_spread": seed_spread,
        "rank_correlation_by_seed": correlations,
        "mean_rank_correlation": mean_corr,
        "max_rank_correlation": max_corr,
        "mean_selection_optimism": mean_optimism,
        "rotation_weight_stats": weight_stats,
        "thresholds": {
            "seed_spread_stop": float(seed_spread_stop),
            "correlation_stop": float(correlation_stop),
            "weight_std_warn": float(weight_std_warn),
            "weight_range_warn": float(weight_range_warn),
            "selection_optimism_warn": float(selection_optimism_warn),
        },
        "stop_flags": stop_flags,
        "warning_flags": warning_flags,
        "status": "STOP_AND_REPORT" if stop_flags else "S3_DIAGNOSTICS_CLEAR",
    }


def resolution_from_promotions(
    diagnostics: dict[str, Any],
    *,
    combined_to_blend_promotion: str | Path | None = None,
    raw_to_combined_promotion: str | Path | None = None,
) -> dict[str, Any]:
    """Turn S3 outputs into an explicit research routing decision."""

    combined_blend = _read_json(combined_to_blend_promotion) if combined_to_blend_promotion is not None else {}
    raw_combined = _read_json(raw_to_combined_promotion) if raw_to_combined_promotion is not None else {}
    blend_verdict = combined_blend.get("verdict")
    representation_verdict = raw_combined.get("verdict")

    if diagnostics.get("stop_flags"):
        route = "STOP_AND_DIAGNOSE"
    elif blend_verdict == "PROMOTED":
        route = "FREEZE_DUAL_VIEW_BACKBONE_AND_RETRY_DIVERSITY"
    elif representation_verdict == "PROMOTED":
        route = "FREEZE_COMBINED_BACKBONE_RAW_HEDGE_NOT_PROMOTED"
    else:
        route = "BACKBONE_NOT_S3_PROMOTED_REOPEN_REPRESENTATION"

    return {
        "route": route,
        "representation_verdict": representation_verdict,
        "dual_view_blend_verdict": blend_verdict,
        "stop_flags": diagnostics.get("stop_flags", []),
        "warning_flags": diagnostics.get("warning_flags", []),
    }
