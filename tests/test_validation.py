import numpy as np
import pandas as pd
import pytest

from s6e8.config import TARGET
from s6e8.validation import (
    CompetitionContractError,
    validate_competition_frames,
    validate_submission,
)


def frames():
    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [20.0, 21.0, 22.0, 23.0],
            "daily_screen_time_hours": [6.0, 7.0, 8.0, 9.0],
            "social_media_hours": [2.0, 3.0, 3.5, 4.0],
            "gaming_hours": [1.0, 2.0, np.nan, 1.5],
            "work_study_hours": [5.0, 4.0, 3.0, 2.0],
            "sleep_hours": [8.0, 7.5, 7.0, 6.5],
            "notifications_per_day": [50.0, 80.0, 120.0, 160.0],
            "app_opens_per_day": [30.0, 45.0, 60.0, 80.0],
            "weekend_screen_time": [7.0, 8.0, 10.0, 11.0],
            "gender": ["Female", "Male", "Female", "Male"],
            "stress_level": ["Low", "Medium", "High", "Medium"],
            "academic_work_impact": ["No", "No", "Yes", "Yes"],
            TARGET: [0, 0, 1, 1],
        }
    )
    test = train.drop(columns=[TARGET]).iloc[:3].copy()
    test["id"] = [10, 11, 12]
    sample = pd.DataFrame({"id": test["id"], TARGET: 0.5})
    return train, test, sample


def test_competition_contract_accepts_expected_frames():
    train, test, sample = frames()
    meta = validate_competition_frames(train, test, sample)
    assert meta["train_rows"] == 4
    assert meta["test_rows"] == 3
    assert meta["target_positive_rate"] == 0.5


def test_competition_contract_rejects_reordered_sample_ids():
    train, test, sample = frames()
    sample = sample.iloc[::-1].reset_index(drop=True)
    with pytest.raises(CompetitionContractError, match="not aligned"):
        validate_competition_frames(train, test, sample)


def test_competition_contract_rejects_duplicate_test_ids():
    train, test, sample = frames()
    test.loc[1, "id"] = test.loc[0, "id"]
    sample["id"] = test["id"]
    with pytest.raises(CompetitionContractError, match="must be unique"):
        validate_competition_frames(train, test, sample)


def test_competition_contract_rejects_infinite_predictor():
    train, test, sample = frames()
    test.loc[0, "age"] = np.inf
    with pytest.raises(CompetitionContractError, match="contains \\+/-inf"):
        validate_competition_frames(train, test, sample)


def test_submission_contract_rejects_nonfinite_scores():
    _, test, sample = frames()
    sub = sample.copy()
    sub[TARGET] = [0.1, np.nan, 0.9]
    with pytest.raises(CompetitionContractError, match="non-finite"):
        validate_submission(sub, test)


def test_submission_contract_allows_arbitrary_finite_ranking_scores():
    _, test, sample = frames()
    sub = sample.copy()
    sub[TARGET] = [-2.0, 0.0, 7.0]
    meta = validate_submission(sub, test)
    assert meta["rows"] == 3
    assert meta["unique_predictions"] == 3
