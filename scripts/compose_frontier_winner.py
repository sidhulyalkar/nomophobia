#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from s6e8.config import ID_COL, TARGET
from s6e8.contrast import rank01
from s6e8.evaluate import delong_test
from s6e8.winner import (
    all_stability_slices_positive,
    compose_rank_score,
    nested_residual_selection,
    stability_diagnostics,
    validate_direction,
)

DEFAULT_LGB_GRID = (0.0, 0.15, 0.20, 0.25, 0.30)
DEFAULT_XGB_GRID = (0.0, 0.05, 0.10, 0.15)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_grid(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None:
        return default
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values or 0.0 not in values:
        raise ValueError("every weight grid must include 0")
    return values


def _load_direction(folder: Path, n_train: int, n_test: int, folds: np.ndarray):
    direction_oof = np.load(folder / "direction_oof.npy")
    direction_test = np.load(folder / "direction_test.npy")
    saved_folds = np.load(folder / "folds.npy")
    return validate_direction(
        direction_oof,
        direction_test,
        folds,
        n_train=n_train,
        n_test=n_test,
        saved_folds=saved_folds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose the evidence-gated frontier winner from the aligned quad anchor "
            "and matched LightGBM/XGBoost identity+screen residual artifacts."
        )
    )
    parser.add_argument("--anchor-oof", required=True)
    parser.add_argument("--anchor-test", required=True)
    parser.add_argument("--lgb-dir", required=True)
    parser.add_argument("--xgb-dir", required=True)
    parser.add_argument("--lgb-grid", default=None)
    parser.add_argument("--xgb-grid", default=None)
    parser.add_argument("--out-dir", default="artifacts/frontier_winner")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(args.anchor_oof)
    test = pd.read_csv(args.anchor_test)
    required_oof = {ID_COL, TARGET, "fold", "honest_blend"}
    if not required_oof.issubset(oof.columns):
        raise ValueError(f"anchor OOF missing {sorted(required_oof - set(oof.columns))}")
    if not {ID_COL, TARGET}.issubset(test.columns):
        raise ValueError("anchor test submission must contain id and target columns")

    y = oof[TARGET].to_numpy(np.int8)
    ids = oof[ID_COL].to_numpy(np.int64)
    folds = oof["fold"].to_numpy(int)
    anchor_oof = rank01(oof["honest_blend"].to_numpy(float))
    anchor_test = rank01(test[TARGET].to_numpy(float))

    lgb_oof, lgb_test = _load_direction(Path(args.lgb_dir), len(oof), len(test), folds)
    xgb_oof, xgb_test = _load_direction(Path(args.xgb_dir), len(oof), len(test), folds)
    directions_oof = {
        "lgb_identity_screen": lgb_oof,
        "xgb_identity_screen": xgb_oof,
    }
    grids = {
        "lgb_identity_screen": _parse_grid(args.lgb_grid, DEFAULT_LGB_GRID),
        "xgb_identity_screen": _parse_grid(args.xgb_grid, DEFAULT_XGB_GRID),
    }

    decision = nested_residual_selection(y, anchor_oof, directions_oof, folds, grids)
    stress = stability_diagnostics(y, anchor_oof, decision["deploy_oof"], ids)
    accepted = bool(
        decision["accepted"]
        and decision["deploy_oof_gain"] > 0
        and all_stability_slices_positive(stress)
    )
    deploy_weights = decision["deploy_weights"] if accepted else {
        "lgb_identity_screen": 0.0,
        "xgb_identity_screen": 0.0,
    }
    deploy_oof = decision["deploy_oof"] if accepted else anchor_oof
    prediction = compose_rank_score(
        anchor_test,
        {
            "lgb_identity_screen": lgb_test,
            "xgb_identity_screen": xgb_test,
        },
        deploy_weights,
        rerank=True,
    )
    if not np.isfinite(prediction).all() or not np.isfinite(deploy_oof).all():
        raise ValueError("frontier winner contains non-finite predictions")

    oof_path = out / "oof_frontier_winner.csv"
    pd.DataFrame(
        {
            ID_COL: ids,
            TARGET: y,
            "fold": folds,
            "frontier_winner": deploy_oof,
        }
    ).to_csv(oof_path, index=False)

    submission = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: prediction})
    submission_path = out / "submission_frontier_winner.csv"
    submission.to_csv(submission_path, index=False)

    serializable_decision = {
        key: value
        for key, value in decision.items()
        if key not in {"honest_oof", "deploy_oof"}
    }
    report = {
        "version": "frontier-residual-winner-v3",
        "accepted": accepted,
        "selection": serializable_decision,
        "honest_delong_p": float(delong_test(y, anchor_oof, decision["honest_oof"])),
        "deploy_weights": deploy_weights,
        "direction_corr_oof": float(np.corrcoef(lgb_oof, xgb_oof)[0, 1]),
        "direction_corr_test": float(np.corrcoef(lgb_test, xgb_test)[0, 1]),
        "stress": stress,
        "oof": {
            "file": oof_path.name,
            "sha256": _sha256(oof_path),
            "auc": float(__import__("sklearn.metrics").metrics.roc_auc_score(y, deploy_oof)),
        },
        "submission": {
            "file": submission_path.name,
            "sha256": _sha256(submission_path),
            "rows": int(len(submission)),
            "unique_predictions": int(submission[TARGET].nunique()),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
        },
    }
    (out / "winner_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
