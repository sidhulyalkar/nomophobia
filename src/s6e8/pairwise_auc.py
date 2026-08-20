from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from .config import CAT_COLS, NUM_COLS, RAW_COLS

_MISSING_CAT = "__missing__"
_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass(frozen=True)
class PairwiseBasis:
    raw_medians: dict[str, float]
    numeric_columns: tuple[str, ...]
    centers: dict[str, float]
    scales: dict[str, float]
    knots: dict[str, tuple[float, ...]]
    categories: dict[str, tuple[str, ...]]
    feature_names: tuple[str, ...]
    output_mean: np.ndarray
    output_scale: np.ndarray


@dataclass(frozen=True)
class PairwiseRanker:
    basis: PairwiseBasis
    coefficients: np.ndarray
    offsets: tuple[int, ...]
    pair_counts: dict[int, int]
    pair_total: int
    alpha: float
    max_iter: int
    seed: int


def _normalized_category(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .fillna(_MISSING_CAT)
    )


def _raw_medians(frame: pd.DataFrame) -> dict[str, float]:
    medians: dict[str, float] = {}
    for column in NUM_COLS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        medians[column] = float(np.median(finite)) if len(finite) else 0.0
    return medians


def _imputed_numeric(frame: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for column in NUM_COLS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        values = np.where(np.isfinite(values), values, medians[column])
        data[column] = values.astype(float, copy=False)
    base = pd.DataFrame(data, index=frame.index)

    daily = base["daily_screen_time_hours"].to_numpy(float)
    social = base["social_media_hours"].to_numpy(float)
    gaming = base["gaming_hours"].to_numpy(float)
    work = base["work_study_hours"].to_numpy(float)
    sleep = base["sleep_hours"].to_numpy(float)
    notifications = base["notifications_per_day"].to_numpy(float)
    opens = base["app_opens_per_day"].to_numpy(float)
    weekend = base["weekend_screen_time"].to_numpy(float)
    component_sum = social + gaming + work
    safe_daily = np.maximum(daily, 0.25)

    derived = {
        "rel__component_sum": component_sum,
        "rel__screen_residual": daily - component_sum,
        "rel__weekend_gap": weekend - daily,
        "rel__daily_social_product": daily * social,
        "rel__daily_weekend_product": daily * weekend,
        "rel__social_weekend_product": social * weekend,
        "rel__social_share": social / safe_daily,
        "rel__gaming_share": gaming / safe_daily,
        "rel__work_share": work / safe_daily,
        "rel__active_share": component_sum / safe_daily,
        "rel__notifications_per_screen": notifications / safe_daily,
        "rel__opens_per_screen": opens / safe_daily,
        "rel__sleep_minus_daily": sleep - daily,
        "rel__weekend_over_daily": weekend / safe_daily,
    }
    for name, values in derived.items():
        base[name] = np.asarray(values, dtype=float)
    return base


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 0.0, 1.0
    center = float(np.median(finite))
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return center, scale


def _unstandardized_basis(
    frame: pd.DataFrame,
    *,
    raw_medians: dict[str, float],
    numeric_columns: tuple[str, ...],
    centers: dict[str, float],
    scales: dict[str, float],
    knots: dict[str, tuple[float, ...]],
    categories: dict[str, tuple[str, ...]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    numeric = _imputed_numeric(frame, raw_medians)
    pieces: list[np.ndarray] = []
    names: list[str] = []

    for column in numeric_columns:
        values = numeric[column].to_numpy(float)
        z = np.clip((values - centers[column]) / scales[column], -8.0, 8.0)
        pieces.append(z[:, None])
        names.append(f"{column}__z")
        pieces.append((z * z)[:, None])
        names.append(f"{column}__z2")
        for idx, knot in enumerate(knots[column]):
            pieces.append(np.maximum(z - float(knot), 0.0)[:, None])
            names.append(f"{column}__hinge{idx}")

    for column in NUM_COLS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        missing = (~np.isfinite(values)).astype(np.float32)
        pieces.append(missing[:, None])
        names.append(f"{column}__missing")

    for column in CAT_COLS:
        values = _normalized_category(frame[column])
        for category in categories[column]:
            pieces.append((values == category).to_numpy(np.float32)[:, None])
            names.append(f"{column}__{category}")

    matrix = np.concatenate(pieces, axis=1).astype(np.float32, copy=False)
    return matrix, tuple(names)


def fit_pairwise_basis(frame: pd.DataFrame) -> PairwiseBasis:
    missing = [column for column in RAW_COLS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing raw feature columns: {missing}")
    medians = _raw_medians(frame)
    numeric = _imputed_numeric(frame, medians)
    numeric_columns = tuple(numeric.columns.tolist())
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    knots: dict[str, tuple[float, ...]] = {}
    for column in numeric_columns:
        values = numeric[column].to_numpy(float)
        center, scale = _robust_center_scale(values)
        z = np.clip((values - center) / scale, -8.0, 8.0)
        centers[column] = center
        scales[column] = scale
        knots[column] = tuple(float(value) for value in np.quantile(z, _QUANTILES))

    categories = {
        column: tuple(sorted(_normalized_category(frame[column]).unique().tolist()))
        for column in CAT_COLS
    }
    raw, feature_names = _unstandardized_basis(
        frame,
        raw_medians=medians,
        numeric_columns=numeric_columns,
        centers=centers,
        scales=scales,
        knots=knots,
        categories=categories,
    )
    output_mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    output_scale = raw.std(axis=0, dtype=np.float64).astype(np.float32)
    output_scale[~np.isfinite(output_scale) | (output_scale < 1e-6)] = 1.0
    return PairwiseBasis(
        raw_medians=medians,
        numeric_columns=numeric_columns,
        centers=centers,
        scales=scales,
        knots=knots,
        categories=categories,
        feature_names=feature_names,
        output_mean=output_mean,
        output_scale=output_scale,
    )


def transform_pairwise_basis(frame: pd.DataFrame, basis: PairwiseBasis) -> np.ndarray:
    raw, names = _unstandardized_basis(
        frame,
        raw_medians=basis.raw_medians,
        numeric_columns=basis.numeric_columns,
        centers=basis.centers,
        scales=basis.scales,
        knots=basis.knots,
        categories=basis.categories,
    )
    if names != basis.feature_names:
        raise RuntimeError("pairwise basis feature contract changed")
    out = (raw - basis.output_mean) / basis.output_scale
    return np.clip(out, -10.0, 10.0).astype(np.float32, copy=False)


def _sample_local_pairs(
    anchor: np.ndarray,
    y: np.ndarray,
    *,
    offsets: tuple[int, ...],
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    anchor = np.asarray(anchor, dtype=float)
    y = np.asarray(y, dtype=np.int8)
    if len(anchor) != len(y):
        raise ValueError("anchor and y must align")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if not offsets or any(offset <= 0 for offset in offsets):
        raise ValueError("offsets must be positive")

    rng = np.random.default_rng(seed)
    order = np.argsort(anchor, kind="mergesort")
    per_offset = max(1, max_pairs // len(offsets))
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    counts: dict[int, int] = {}

    for offset in offsets:
        if offset >= len(order):
            counts[int(offset)] = 0
            continue
        left = order[:-offset]
        right = order[offset:]
        mask = y[left] != y[right]
        left = left[mask]
        right = right[mask]
        available = len(left)
        if available > per_offset:
            selected = np.sort(rng.choice(available, size=per_offset, replace=False))
            left = left[selected]
            right = right[selected]
        pos = np.where(y[left] == 1, left, right)
        neg = np.where(y[left] == 0, left, right)
        positives.append(pos.astype(np.int64, copy=False))
        negatives.append(neg.astype(np.int64, copy=False))
        local_weight = np.full(len(pos), 1.0 / np.sqrt(float(offset)), dtype=np.float32)
        weights.append(local_weight)
        counts[int(offset)] = int(len(pos))

    if not positives or sum(len(values) for values in positives) < 1000:
        raise ValueError("not enough local opposite-class pairs")
    pos = np.concatenate(positives)
    neg = np.concatenate(negatives)
    sample_weight = np.concatenate(weights)
    sample_weight /= float(sample_weight.mean())
    return pos, neg, sample_weight, counts


def fit_pairwise_ranker(
    frame: pd.DataFrame,
    y: np.ndarray,
    anchor: np.ndarray,
    *,
    offsets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    max_pairs: int = 210_000,
    alpha: float = 1e-4,
    max_iter: int = 50,
    seed: int = 20260819,
) -> PairwiseRanker:
    """Fit a linear AUC correction on hard positive/negative anchor pairs.

    The model never receives the anchor as a predictive feature.  The anchor is
    used only to choose nearby opposite-class training pairs, so the learned
    score answers a narrower question: among rows the champion nearly ties,
    which raw-feature pattern should rank higher?
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    y = np.asarray(y, dtype=np.int8)
    anchor = np.asarray(anchor, dtype=float)
    if len(frame) != len(y) or len(y) != len(anchor):
        raise ValueError("frame, y, and anchor must align")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("y must contain both binary classes")

    basis = fit_pairwise_basis(frame)
    matrix = transform_pairwise_basis(frame, basis)
    pos, neg, weights, pair_counts = _sample_local_pairs(
        anchor,
        y,
        offsets=offsets,
        max_pairs=max_pairs,
        seed=seed,
    )
    diff = matrix[pos] - matrix[neg]
    design = np.concatenate([diff, -diff], axis=0)
    labels = np.concatenate(
        [np.ones(len(diff), dtype=np.int8), np.zeros(len(diff), dtype=np.int8)]
    )
    fit_weights = np.concatenate([weights, weights])

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(alpha),
        fit_intercept=False,
        max_iter=int(max_iter),
        tol=1e-4,
        shuffle=True,
        random_state=int(seed),
        average=True,
    )
    classifier.fit(design, labels, sample_weight=fit_weights)
    coefficients = classifier.coef_[0].astype(np.float32, copy=True)
    if not np.isfinite(coefficients).all():
        raise RuntimeError("pairwise ranker produced non-finite coefficients")
    return PairwiseRanker(
        basis=basis,
        coefficients=coefficients,
        offsets=tuple(int(value) for value in offsets),
        pair_counts=pair_counts,
        pair_total=int(len(diff)),
        alpha=float(alpha),
        max_iter=int(max_iter),
        seed=int(seed),
    )


def score_pairwise_ranker(model: PairwiseRanker, frame: pd.DataFrame) -> np.ndarray:
    matrix = transform_pairwise_basis(frame, model.basis)
    score = matrix @ model.coefficients
    if not np.isfinite(score).all():
        raise RuntimeError("pairwise ranker score contains non-finite values")
    return np.asarray(score, dtype=float)


def pairwise_ranker_report(model: PairwiseRanker, *, top_n: int = 20) -> dict:
    coef = np.asarray(model.coefficients, dtype=float)
    order = np.argsort(np.abs(coef))[::-1][:top_n]
    return {
        "feature_count": len(model.basis.feature_names),
        "pair_total": model.pair_total,
        "pair_counts": {str(key): value for key, value in model.pair_counts.items()},
        "offsets": list(model.offsets),
        "alpha": model.alpha,
        "max_iter": model.max_iter,
        "seed": model.seed,
        "top_coefficients": [
            {
                "feature": model.basis.feature_names[int(idx)],
                "coefficient": float(coef[int(idx)]),
            }
            for idx in order
        ],
    }
