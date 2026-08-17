from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import CAT_COLS, ID_COL, NUM_COLS, RAW_COLS, TARGET


class CompetitionContractError(ValueError):
    """Raised when Kaggle inputs or outputs violate the expected competition contract."""


def _require_columns(
    frame: pd.DataFrame,
    expected: Iterable[str],
    *,
    name: str,
    exact: bool = True,
) -> None:
    expected = list(expected)
    missing = [c for c in expected if c not in frame.columns]
    extra = [c for c in frame.columns if c not in expected] if exact else []
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"unexpected={extra}")
        raise CompetitionContractError(f"{name} schema mismatch: " + ", ".join(parts))


def _validate_ids(frame: pd.DataFrame, *, name: str) -> np.ndarray:
    ids = frame[ID_COL]
    if ids.isna().any():
        raise CompetitionContractError(f"{name}.{ID_COL} contains missing values")
    if ids.duplicated().any():
        duplicates = ids[ids.duplicated(keep=False)].head(5).tolist()
        raise CompetitionContractError(
            f"{name}.{ID_COL} must be unique; example duplicates={duplicates}"
        )
    return ids.to_numpy(copy=False)


def _validate_numeric_predictors(frame: pd.DataFrame, *, name: str) -> None:
    for column in NUM_COLS:
        converted = pd.to_numeric(frame[column], errors="coerce")
        unparsable = frame[column].notna() & converted.isna()
        if unparsable.any():
            examples = frame.loc[unparsable, column].head(5).tolist()
            raise CompetitionContractError(
                f"{name}.{column} contains non-numeric values: {examples}"
            )
        values = converted.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise CompetitionContractError(f"{name}.{column} contains +/-inf")


def _validate_target(target: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(target, errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise CompetitionContractError(f"train.{TARGET} must be finite and non-missing")
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0.0, 1.0}) or len(unique) != 2:
        raise CompetitionContractError(
            f"train.{TARGET} must contain both binary classes 0 and 1; found={sorted(unique)}"
        )
    return values.astype(np.int8)


def validate_competition_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    *,
    exact_schema: bool = True,
) -> dict:
    """Validate the three Kaggle CSVs before feature engineering or model fitting.

    The most important invariant is ID alignment: sample submission IDs must match test IDs
    in the same order. A reordered submission can look perfectly valid to pandas while being
    useless on the leaderboard, so this check intentionally fails hard.
    """

    train_columns = [ID_COL, *RAW_COLS, TARGET]
    test_columns = [ID_COL, *RAW_COLS]
    sample_columns = [ID_COL, TARGET]
    _require_columns(train, train_columns, name="train", exact=exact_schema)
    _require_columns(test, test_columns, name="test", exact=exact_schema)
    _require_columns(
        sample_submission, sample_columns, name="sample_submission", exact=exact_schema
    )

    train_ids = _validate_ids(train, name="train")
    test_ids = _validate_ids(test, name="test")
    sample_ids = _validate_ids(sample_submission, name="sample_submission")
    if len(test_ids) != len(sample_ids):
        raise CompetitionContractError(
            f"sample_submission rows ({len(sample_ids)}) != test rows ({len(test_ids)})"
        )
    if not np.array_equal(test_ids, sample_ids):
        mismatch = np.flatnonzero(test_ids != sample_ids)
        pos = int(mismatch[0]) if len(mismatch) else 0
        raise CompetitionContractError(
            "sample_submission IDs are not aligned to test IDs in row order; "
            f"first mismatch at row {pos}: test={test_ids[pos]!r}, sample={sample_ids[pos]!r}"
        )

    _validate_numeric_predictors(train, name="train")
    _validate_numeric_predictors(test, name="test")
    y = _validate_target(train[TARGET])

    for column in CAT_COLS:
        _ = train[column].astype("string")
        _ = test[column].astype("string")

    return {
        "train_rows": int(len(train_ids)),
        "test_rows": int(len(test_ids)),
        "train_columns": int(train.shape[1]),
        "test_columns": int(test.shape[1]),
        "target_positive_rate": float(y.mean()),
    }


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    *,
    require_nonconstant: bool = True,
) -> dict:
    """Validate a Kaggle prediction frame without constraining scores to [0, 1]."""

    _require_columns(submission, [ID_COL, TARGET], name="submission", exact=True)
    _require_columns(test, [ID_COL, *RAW_COLS], name="test", exact=False)
    sub_ids = _validate_ids(submission, name="submission")
    test_ids = _validate_ids(test, name="test")
    if len(sub_ids) != len(test_ids):
        raise CompetitionContractError(
            f"submission rows ({len(sub_ids)}) != test rows ({len(test_ids)})"
        )
    if not np.array_equal(sub_ids, test_ids):
        mismatch = np.flatnonzero(sub_ids != test_ids)
        pos = int(mismatch[0]) if len(mismatch) else 0
        raise CompetitionContractError(
            "submission IDs do not exactly match test IDs in row order; "
            f"first mismatch at row {pos}: submission={sub_ids[pos]!r}, test={test_ids[pos]!r}"
        )

    numeric = pd.to_numeric(submission[TARGET], errors="coerce")
    score = numeric.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(score).all():
        bad = np.flatnonzero(~np.isfinite(score))[:5].tolist()
        raise CompetitionContractError(
            f"submission.{TARGET} contains non-finite values at rows {bad}"
        )
    unique = int(pd.Series(score).nunique(dropna=False))
    if require_nonconstant and len(score) > 1 and unique < 2:
        raise CompetitionContractError(f"submission.{TARGET} is constant")

    return {
        "rows": int(len(score)),
        "unique_predictions": unique,
        "prediction_min": float(score.min()) if len(score) else None,
        "prediction_max": float(score.max()) if len(score) else None,
        "prediction_mean": float(score.mean()) if len(score) else None,
        "prediction_std": float(score.std()) if len(score) else None,
    }
