from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CAT_COLS, NUM_COLS


@dataclass(frozen=True)
class TokenSpec:
    """A target-free row identity token used by the generator-evidence graph."""

    name: str
    columns: tuple[str, ...]
    decimals: tuple[int | None, ...]

    def validate(self) -> None:
        if not self.columns or len(self.columns) != len(self.decimals):
            raise ValueError("TokenSpec columns/decimals must be non-empty and aligned")


EXACT_SPECS: tuple[TokenSpec, ...] = tuple(
    TokenSpec(f"exact__{column}", (column,), (None,)) for column in NUM_COLS
)

# The joint tokens deliberately encode plausible latent source-row geometry rather
# than an indiscriminate all-pairs explosion.  Exact univariate identity is the
# matched control; these interactions are the structural treatment.
JOINT_SPECS: tuple[TokenSpec, ...] = (
    TokenSpec("joint__daily_social", ("daily_screen_time_hours", "social_media_hours"), (2, 2)),
    TokenSpec("joint__daily_weekend", ("daily_screen_time_hours", "weekend_screen_time"), (2, 2)),
    TokenSpec("joint__daily_work", ("daily_screen_time_hours", "work_study_hours"), (2, 2)),
    TokenSpec("joint__social_gaming", ("social_media_hours", "gaming_hours"), (2, 2)),
    TokenSpec("joint__sleep_daily", ("sleep_hours", "daily_screen_time_hours"), (2, 2)),
    TokenSpec("joint__notifications_opens", ("notifications_per_day", "app_opens_per_day"), (0, 0)),
    TokenSpec("joint__age_daily", ("age", "daily_screen_time_hours"), (0, 1)),
    TokenSpec("joint__age_weekend", ("age", "weekend_screen_time"), (0, 1)),
    TokenSpec("joint__stress_daily", ("stress_level", "daily_screen_time_hours"), (None, 1)),
    TokenSpec("joint__impact_work", ("academic_work_impact", "work_study_hours"), (None, 1)),
    TokenSpec("joint__gender_social", ("gender", "social_media_hours"), (None, 1)),
)


def default_treatment_specs() -> tuple[TokenSpec, ...]:
    return EXACT_SPECS + JOINT_SPECS


def _transform_column(series: pd.Series, decimals: int | None) -> pd.Series:
    if series.name in CAT_COLS:
        return series.astype("string").fillna("__MISSING__")
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    if decimals is not None:
        values = values.round(decimals)
    return values


def token_hash(frame: pd.DataFrame, spec: TokenSpec) -> np.ndarray:
    """Return a deterministic uint64 token without materializing giant strings."""
    spec.validate()
    missing = [column for column in spec.columns if column not in frame]
    if missing:
        raise ValueError(f"missing token columns for {spec.name}: {missing}")
    transformed = pd.DataFrame(
        {
            f"c{i}": _transform_column(frame[column], decimals)
            for i, (column, decimals) in enumerate(zip(spec.columns, spec.decimals))
        },
        index=frame.index,
    )
    return pd.util.hash_pandas_object(transformed, index=False).to_numpy(np.uint64)


def _posterior_table(
    keys: np.ndarray,
    y: np.ndarray,
    *,
    prior: float,
    smoothing: float,
) -> pd.DataFrame:
    table = pd.DataFrame({"key": keys, "y": np.asarray(y, dtype=np.float64)})
    stats = table.groupby("key", sort=False, observed=True)["y"].agg(["sum", "count"])
    stats["posterior"] = (stats["sum"] + smoothing * prior) / (
        stats["count"] + smoothing
    )
    return stats[["posterior", "count"]]


def graph_score(
    reference: pd.DataFrame,
    y_reference: np.ndarray,
    query: pd.DataFrame,
    specs: Iterable[TokenSpec],
    *,
    smoothing: float = 10.0,
    min_support: int = 2,
) -> tuple[np.ndarray, dict[str, float]]:
    """Score query rows from label evidence attached to target-free identity tokens.

    Labels only enter through ``reference``.  Each token posterior is empirically
    shrunk to the reference prior and weighted by its effective support.  Query
    labels are never inspected, making this safe for outer-fold validation/test.
    """
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    y = np.asarray(y_reference, dtype=np.float64)
    if len(reference) != len(y):
        raise ValueError("reference and y_reference must align")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("y_reference must be binary")

    prior = float(np.mean(y))
    numerator = np.zeros(len(query), dtype=np.float64)
    denominator = np.zeros(len(query), dtype=np.float64)
    matched_tokens = np.zeros(len(query), dtype=np.int16)
    family_coverages: dict[str, float] = {}

    for spec in specs:
        ref_keys = token_hash(reference, spec)
        query_keys = token_hash(query, spec)
        stats = _posterior_table(ref_keys, y, prior=prior, smoothing=smoothing)
        posterior_map = stats["posterior"]
        count_map = stats["count"]

        query_series = pd.Series(query_keys, index=query.index)
        posterior = query_series.map(posterior_map).to_numpy(np.float64, na_value=np.nan)
        support = query_series.map(count_map).to_numpy(np.float64, na_value=np.nan)
        seen = np.isfinite(posterior) & np.isfinite(support) & (support >= min_support)

        reliability = np.zeros(len(query), dtype=np.float64)
        reliability[seen] = support[seen] / (support[seen] + smoothing)
        values = np.full(len(query), prior, dtype=np.float64)
        values[seen] = posterior[seen]
        numerator += reliability * values
        denominator += reliability
        matched_tokens += seen.astype(np.int16)
        family_coverages[spec.name] = float(np.mean(seen))

    score = np.full(len(query), prior, dtype=np.float64)
    has_evidence = denominator > 0
    score[has_evidence] = numerator[has_evidence] / denominator[has_evidence]
    diagnostics = {
        "prior": prior,
        "coverage_any": float(np.mean(has_evidence)),
        "mean_matched_tokens": float(np.mean(matched_tokens)),
        "mean_effective_weight": float(np.mean(denominator)),
        **{f"coverage::{key}": value for key, value in family_coverages.items()},
    }
    return score, diagnostics


def cross_fitted_graph_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    control_specs: Iterable[TokenSpec] = EXACT_SPECS,
    treatment_specs: Iterable[TokenSpec] | None = None,
    smoothing: float = 10.0,
    min_support: int = 2,
) -> dict[str, object]:
    """Build aligned control/treatment OOF scores and fold-bagged test scores."""
    y = np.asarray(y, dtype=np.int8)
    folds = np.asarray(folds)
    if not (len(train) == len(y) == len(folds)):
        raise ValueError("train, y, and folds must align")
    treatment_specs = tuple(treatment_specs or default_treatment_specs())
    control_specs = tuple(control_specs)
    unique_folds = np.unique(folds)
    if len(unique_folds) < 3:
        raise ValueError("at least three outer folds are required")

    oof_control = np.empty(len(train), dtype=np.float64)
    oof_treatment = np.empty(len(train), dtype=np.float64)
    test_control = np.zeros(len(test), dtype=np.float64)
    test_treatment = np.zeros(len(test), dtype=np.float64)
    fold_diagnostics: list[dict[str, object]] = []

    for fold in unique_folds:
        reference_mask = folds != fold
        query_mask = folds == fold
        reference = train.loc[reference_mask]
        query = train.loc[query_mask]
        yy = y[reference_mask]

        control_valid, control_diag = graph_score(
            reference,
            yy,
            query,
            control_specs,
            smoothing=smoothing,
            min_support=min_support,
        )
        treatment_valid, treatment_diag = graph_score(
            reference,
            yy,
            query,
            treatment_specs,
            smoothing=smoothing,
            min_support=min_support,
        )
        control_test, _ = graph_score(
            reference,
            yy,
            test,
            control_specs,
            smoothing=smoothing,
            min_support=min_support,
        )
        treatment_test, _ = graph_score(
            reference,
            yy,
            test,
            treatment_specs,
            smoothing=smoothing,
            min_support=min_support,
        )

        oof_control[query_mask] = control_valid
        oof_treatment[query_mask] = treatment_valid
        test_control += control_test / len(unique_folds)
        test_treatment += treatment_test / len(unique_folds)
        fold_diagnostics.append(
            {
                "fold": int(fold),
                "control": control_diag,
                "treatment": treatment_diag,
            }
        )

    return {
        "oof_control": oof_control,
        "oof_treatment": oof_treatment,
        "test_control": test_control,
        "test_treatment": test_treatment,
        "fold_diagnostics": fold_diagnostics,
        "control_specs": [spec.name for spec in control_specs],
        "treatment_specs": [spec.name for spec in treatment_specs],
    }
