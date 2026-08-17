from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .artifacts import sha256_file
from .config import ID_COL, TARGET
from .validation import CompetitionContractError, validate_submission


def unit_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise CompetitionContractError("rank input must be a non-empty 1D array")
    if not np.isfinite(values).all():
        raise CompetitionContractError("rank input contains non-finite predictions")
    return (rankdata(values, method="average") - 0.5) / len(values)


def rank_blend(
    predictions: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Blend independently ranked prediction streams with explicit convex weights."""

    if not predictions:
        raise CompetitionContractError("at least one prediction stream is required")
    if set(predictions) != set(weights):
        raise CompetitionContractError(
            "prediction and weight names must match exactly; "
            f"predictions={sorted(predictions)}, weights={sorted(weights)}"
        )
    names = list(predictions)
    w = np.asarray([weights[name] for name in names], dtype=float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise CompetitionContractError("blend weights must be finite and non-negative")
    if not np.isclose(w.sum(), 1.0, atol=1e-10):
        raise CompetitionContractError(f"blend weights must sum to 1; got {w.sum():.12g}")

    ranked = []
    expected = None
    for name in names:
        values = np.asarray(predictions[name], dtype=float)
        if expected is None:
            expected = len(values)
        if values.ndim != 1 or len(values) != expected:
            raise CompetitionContractError(
                f"prediction stream {name!r} has incompatible shape {values.shape}"
            )
        ranked.append(unit_rank(values))
    return np.column_stack(ranked) @ w


def build_submission(
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    score: np.ndarray,
) -> pd.DataFrame:
    score = np.asarray(score, dtype=float)
    if score.ndim != 1 or len(score) != len(test):
        raise CompetitionContractError(
            f"prediction length {len(score)} does not match test rows {len(test)}"
        )
    submission = sample_submission[[ID_COL, TARGET]].copy()
    submission[TARGET] = score
    validate_submission(submission, test)
    return submission


def write_submission(
    path: str | Path,
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    score: np.ndarray,
    *,
    verify_roundtrip: bool = True,
) -> dict:
    """Write, re-read, validate, and hash a Kaggle submission artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission = build_submission(sample_submission, test, score)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp = Path(handle.name)
    try:
        with handle:
            submission.to_csv(handle, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

    if verify_roundtrip:
        roundtrip = pd.read_csv(path)
        stats = validate_submission(roundtrip, test)
    else:
        stats = validate_submission(submission, test)
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        **stats,
    }
