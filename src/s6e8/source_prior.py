from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from .config import RAW_COLS
from .preprocess import prepare_tree_frames
from .utils import stable_seed

SEVERITY_ORDER = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}


@dataclass
class SourcePrior:
    train_prob: np.ndarray
    test_prob: np.ndarray
    train_expected_severity: np.ndarray
    test_expected_severity: np.ndarray
    train_entropy: np.ndarray
    test_entropy: np.ndarray
    train_class_prob: np.ndarray
    test_class_prob: np.ndarray
    classes_: list[int]


def _entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1)


def fit_source_ordinal_prior(
    original: pd.DataFrame,
    comp_train: pd.DataFrame,
    comp_test: pd.DataFrame,
    *,
    seed: int = 20260816,
    n_estimators: int = 350,
) -> SourcePrior:
    """Train a source-only four-level severity model and apply it to competition rows."""
    missing = [c for c in RAW_COLS + ["addiction_level"] if c not in original.columns]
    if missing:
        raise ValueError(f"Original dataset is missing required columns: {missing}")
    src = original.reset_index(drop=True).copy()
    level = src["addiction_level"].astype("string").fillna("None")
    y = level.map(SEVERITY_ORDER)
    if y.isna().any():
        bad = sorted(level.loc[y.isna()].astype(str).unique())
        raise ValueError(f"Unknown addiction_level values: {bad}")
    target = pd.concat([
        comp_train[RAW_COLS].reset_index(drop=True),
        comp_test[RAW_COLS].reset_index(drop=True),
    ], ignore_index=True)
    _, _, src_native, target_native, cats = prepare_tree_frames(
        src[RAW_COLS].reset_index(drop=True), target
    )
    model = LGBMClassifier(
        objective="multiclass", num_class=4, n_estimators=n_estimators,
        learning_rate=0.035, num_leaves=31, min_child_samples=40,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
        random_state=stable_seed(seed, "source_ordinal"), n_jobs=-1, verbosity=-1,
    )
    model.fit(src_native, y.astype(int), categorical_feature=cats)
    rawp = model.predict_proba(target_native)
    p = np.zeros((len(target_native), 4), dtype=float)
    for j, cls in enumerate(model.classes_):
        p[:, int(cls)] = rawp[:, j]
    ntr = len(comp_train); levels = np.arange(4, dtype=float); expected = p @ levels
    addicted = p[:, 2] + p[:, 3]; ent = _entropy(p)
    return SourcePrior(
        train_prob=addicted[:ntr], test_prob=addicted[ntr:],
        train_expected_severity=expected[:ntr], test_expected_severity=expected[ntr:],
        train_entropy=ent[:ntr], test_entropy=ent[ntr:],
        train_class_prob=p[:ntr], test_class_prob=p[ntr:], classes_=[0,1,2,3],
    )


def clipped_logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    q = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(q / (1 - q))
