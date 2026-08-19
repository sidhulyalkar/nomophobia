from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import CAT_COLS, RAW_COLS

_MISSING_CAT = "__missing__"


@dataclass(frozen=True)
class DomainShiftResult:
    train_probability: np.ndarray
    test_probability: np.ndarray
    train_importance_weight: np.ndarray
    domain_auc: float
    fold_aucs: tuple[float, ...]
    raw_weight_min: float
    raw_weight_max: float
    clipped_weight_min: float
    clipped_weight_max: float
    effective_sample_size: float


def _canonical_series(values: pd.Series, *, categorical: bool) -> pd.Series:
    if categorical:
        return values.astype("string").str.strip().str.lower().fillna(_MISSING_CAT)
    numeric = pd.to_numeric(values, errors="coerce")
    # String conversion is only used for pooled frequency lookup; keep a stable
    # explicit token for missing values.
    return numeric.map(lambda value: _MISSING_CAT if pd.isna(value) else repr(float(value)))


def build_domain_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    missing_train = [column for column in RAW_COLS if column not in train.columns]
    missing_test = [column for column in RAW_COLS if column not in test.columns]
    if missing_train or missing_test:
        raise ValueError(f"missing raw columns: train={missing_train}, test={missing_test}")

    combined = pd.concat([train[RAW_COLS], test[RAW_COLS]], ignore_index=True)
    out = pd.DataFrame(index=combined.index)
    for column in RAW_COLS:
        if column in CAT_COLS:
            normalized = _canonical_series(combined[column], categorical=True)
            categories = sorted(normalized.unique().tolist())
            out[column] = pd.Categorical(normalized, categories=categories)
            key = normalized
        else:
            out[column] = pd.to_numeric(combined[column], errors="coerce").astype(float)
            key = _canonical_series(combined[column], categorical=False)
        freq = key.map(key.value_counts(dropna=False) / len(key)).to_numpy(np.float32)
        out[f"domain_freq__{column}"] = freq
        out[f"domain_missing__{column}"] = combined[column].isna().to_numpy(np.uint8)

    out["domain_missing_count"] = combined[RAW_COLS].isna().sum(axis=1).to_numpy(np.int16)
    return out


def weighted_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if not (len(y) == len(score) == len(weight)):
        raise ValueError("weighted_auc inputs must align")
    if not np.isfinite(score).all() or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("weighted_auc requires finite scores and non-negative weights")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2:
        raise ValueError("weighted_auc requires both binary classes")

    order = np.argsort(score, kind="mergesort")
    ys = y[order]
    scores = score[order]
    weights = weight[order]
    total_pos = float(weights[ys == 1].sum())
    total_neg = float(weights[ys == 0].sum())
    if total_pos <= 0 or total_neg <= 0:
        raise ValueError("both classes require positive total weight")

    contribution = 0.0
    cumulative_neg = 0.0
    start = 0
    n = len(y)
    while start < n:
        end = start + 1
        while end < n and scores[end] == scores[start]:
            end += 1
        group_y = ys[start:end]
        group_w = weights[start:end]
        pos_w = float(group_w[group_y == 1].sum())
        neg_w = float(group_w[group_y == 0].sum())
        contribution += pos_w * (cumulative_neg + 0.5 * neg_w)
        cumulative_neg += neg_w
        start = end
    return contribution / (total_pos * total_neg)


def fit_domain_shift_weights(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 20260819,
    estimators: int = 350,
    learning_rate: float = 0.05,
    clip_low: float = 0.25,
    clip_high: float = 4.0,
) -> DomainShiftResult:
    if folds < 3:
        raise ValueError("at least three domain folds are required")
    if not 0 < clip_low < 1 < clip_high:
        raise ValueError("importance-weight clipping must straddle one")

    frame = build_domain_frame(train, test)
    n_train = len(train)
    domain = np.concatenate(
        [np.zeros(n_train, dtype=np.int8), np.ones(len(test), dtype=np.int8)]
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.empty(len(frame), dtype=float)
    fold_aucs = []
    categorical = [column for column in CAT_COLS if column in frame.columns]

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(frame, domain)):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=int(estimators),
            learning_rate=float(learning_rate),
            num_leaves=31,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=seed + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            frame.iloc[fit_idx],
            domain[fit_idx],
            categorical_feature=categorical,
        )
        pred = model.predict_proba(frame.iloc[valid_idx])[:, 1]
        oof[valid_idx] = pred
        fold_aucs.append(float(roc_auc_score(domain[valid_idx], pred)))

    domain_auc = float(roc_auc_score(domain, oof))
    p = np.clip(oof[:n_train], 1e-5, 1 - 1e-5)
    prior_train = n_train / len(frame)
    prior_test = len(test) / len(frame)
    raw = (p / (1.0 - p)) * (prior_train / prior_test)
    clipped = np.clip(raw, clip_low, clip_high)
    clipped /= float(np.mean(clipped))
    effective_sample_size = float((clipped.sum() ** 2) / np.square(clipped).sum())
    return DomainShiftResult(
        train_probability=oof[:n_train].copy(),
        test_probability=oof[n_train:].copy(),
        train_importance_weight=clipped.astype(np.float64),
        domain_auc=domain_auc,
        fold_aucs=tuple(fold_aucs),
        raw_weight_min=float(raw.min()),
        raw_weight_max=float(raw.max()),
        clipped_weight_min=float(clipped.min()),
        clipped_weight_max=float(clipped.max()),
        effective_sample_size=effective_sample_size,
    )
