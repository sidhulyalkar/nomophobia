from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from .config import EXPERT_SPECS, TARGET
from .features import build_feature_views
from .io import load_competition
from .models import make_lgb
from .preprocess import prepare_tree_frames
from .utils import stable_seed


@dataclass(frozen=True)
class IterationTuningConfig:
    data_dir: str | Path = "data"
    expert: str = "lgb_combined63"
    rows: int = 628_000
    max_estimators: int = 4_000
    patience: int = 200
    device: str = "gpu"
    seed: int = 20260816
    repeats: int = 3
    validation_fraction: float = 0.12
    ceiling_fraction: float = 0.90


def _stratified_sample(df, rows: int, seed: int):
    if rows >= len(df):
        return df.reset_index(drop=True)
    if rows <= 0:
        raise ValueError("rows must be positive")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=rows, random_state=seed)
    idx, _ = next(splitter.split(np.zeros(len(df), dtype=np.int8), df[TARGET].to_numpy()))
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def _validate_config(config: IterationTuningConfig) -> None:
    if config.expert not in EXPERT_SPECS:
        raise ValueError(f"unknown expert: {config.expert}")
    if EXPERT_SPECS[config.expert]["family"] != "lgb":
        raise ValueError("robust iteration tuning currently supports LightGBM experts only")
    if config.device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu")
    if config.max_estimators <= 0 or config.patience <= 0 or config.repeats <= 0:
        raise ValueError("max_estimators, patience, and repeats must be positive")
    if not 0.02 <= config.validation_fraction <= 0.40:
        raise ValueError("validation_fraction must be between 0.02 and 0.40")
    if not 0.50 <= config.ceiling_fraction <= 1.0:
        raise ValueError("ceiling_fraction must be between 0.50 and 1.0")


def tune_lightgbm_iterations(config: IterationTuningConfig) -> dict[str, Any]:
    """Tune a frozen estimator count using repeated production-scale inner holdouts."""

    _validate_config(config)
    train, test, _ = load_competition(config.data_dir)
    sample = _stratified_sample(train, min(config.rows, len(train)), config.seed)
    y = sample[TARGET].astype(int).reset_index(drop=True)
    spec = EXPERT_SPECS[config.expert]
    views = build_feature_views(sample.drop(columns=[TARGET]), test.iloc[:1].copy(), use_frequency=True, frequency_reference=(train.drop(columns=[TARGET]), test))
    X, _ = views[spec["view"]]

    repeats = []
    best_iterations = []
    best_scores = []
    for repeat in range(config.repeats):
        split_seed = stable_seed(config.seed, "iteration_tuning_split", config.expert, repeat)
        train_idx, valid_idx = train_test_split(np.arange(len(sample)), test_size=config.validation_fraction, stratify=y, random_state=split_seed)
        _, _, X_train, X_valid, cats = prepare_tree_frames(X.iloc[train_idx].reset_index(drop=True), X.iloc[valid_idx].reset_index(drop=True))
        model_seed = stable_seed(config.seed, "iteration_tuning_model", config.expert, repeat)
        model = make_lgb(model_seed, config.max_estimators, spec["profile"], device=config.device)
        model.fit(X_train, y.iloc[train_idx].reset_index(drop=True), eval_set=[(X_valid, y.iloc[valid_idx].reset_index(drop=True))], eval_metric="auc", categorical_feature=cats, callbacks=[lgb.early_stopping(config.patience, verbose=False)])
        best_iteration = int(model.best_iteration_)
        best_score = float(model.best_score_["valid_0"]["auc"])
        best_iterations.append(best_iteration); best_scores.append(best_score)
        repeats.append({"repeat": repeat, "split_seed": int(split_seed), "model_seed": int(model_seed), "training_rows": int(len(train_idx)), "validation_rows": int(len(valid_idx)), "best_iteration": best_iteration, "best_score": best_score, "ceiling_hit": bool(best_iteration >= int(np.ceil(config.ceiling_fraction * config.max_estimators)))})

    selected = int(np.rint(np.median(best_iterations)))
    median = max(float(np.median(best_iterations)), 1.0)
    spread_fraction = float((max(best_iterations) - min(best_iterations)) / median)
    q25, q75 = np.quantile(best_iterations, [0.25, 0.75])
    any_ceiling = any(row["ceiling_hit"] for row in repeats)
    return {"version": "nomophobia-v0.3", "method": "repeated_production_scale_inner_holdout", "expert": config.expert, "rows_total": int(len(sample)), "device": config.device, "max_estimators": int(config.max_estimators), "patience": int(config.patience), "validation_fraction": float(config.validation_fraction), "repeats": int(config.repeats), "repeat_results": repeats, "best_iterations": [int(v) for v in best_iterations], "best_scores": [float(v) for v in best_scores], "best_iteration": selected, "selected_iteration_rule": "median", "iteration_iqr": [float(q25), float(q75)], "iteration_spread_fraction": spread_fraction, "tuning_instability_warning": bool(spread_fraction > 0.30), "ceiling_fraction": float(config.ceiling_fraction), "ceiling_hit_any_repeat": bool(any_ceiling), "ceiling_90pct_hit": bool(any_ceiling and np.isclose(config.ceiling_fraction, 0.90)), "estimator_count": selected, "note": "Freeze this count for all OOF folds; inner tuning AUC is diagnostic only."}
