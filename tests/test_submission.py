import numpy as np
import pandas as pd
import pytest

from s6e8.config import TARGET
from s6e8.submission import rank_blend, unit_rank, write_submission
from s6e8.validation import CompetitionContractError


def test_unit_rank_preserves_ties_and_range():
    score = unit_rank(np.array([10.0, 10.0, 20.0, -5.0]))
    assert np.isclose(score[0], score[1])
    assert score[3] < score[0] < score[2]
    assert 0.0 < score.min() < score.max() < 1.0


def test_rank_blend_matches_manual_two_view_mix():
    a = np.array([0.1, 0.8, 0.3, 0.4])
    b = np.array([0.8, 0.1, 0.4, 0.3])
    got = rank_blend({"a": a, "b": b}, {"a": 0.625, "b": 0.375})
    expected = 0.625 * unit_rank(a) + 0.375 * unit_rank(b)
    assert np.allclose(got, expected)


def test_rank_blend_rejects_accidental_nonconvex_weights():
    with pytest.raises(CompetitionContractError, match="sum to 1"):
        rank_blend(
            {"a": np.array([0.1, 0.2]), "b": np.array([0.2, 0.1])},
            {"a": 0.7, "b": 0.7},
        )


def test_write_submission_roundtrips_and_hashes(tmp_path):
    test = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "age": [20.0, 21.0, 22.0],
            "daily_screen_time_hours": [6.0, 7.0, 8.0],
            "social_media_hours": [2.0, 3.0, 4.0],
            "gaming_hours": [1.0, 2.0, 3.0],
            "work_study_hours": [5.0, 4.0, 3.0],
            "sleep_hours": [8.0, 7.0, 6.0],
            "notifications_per_day": [50.0, 80.0, 120.0],
            "app_opens_per_day": [30.0, 45.0, 60.0],
            "weekend_screen_time": [7.0, 8.0, 10.0],
            "gender": ["Female", "Male", "Female"],
            "stress_level": ["Low", "Medium", "High"],
            "academic_work_impact": ["No", "No", "Yes"],
        }
    )
    sample = pd.DataFrame({"id": test["id"], TARGET: 0.5})
    path = tmp_path / "submission.csv"
    meta = write_submission(path, sample, test, np.array([0.2, 0.4, 0.8]))
    assert path.exists()
    assert meta["rows"] == 3
    assert len(meta["sha256"]) == 64
    roundtrip = pd.read_csv(path)
    assert roundtrip["id"].tolist() == [10, 11, 12]


def test_frequency_ablation_groups_are_mutually_classified():
    from s6e8.frequency import frequency_columns_for_arm, frequency_feature_groups

    columns = [
        "age",
        "age__freq",
        "age__logfreq",
        "daily_screen_time_hours__round0_freq",
        "gender__freq",
        "gender__logfreq",
    ]
    groups = frequency_feature_groups(columns)
    flat = [c for values in groups.values() for c in values]
    assert len(flat) == len(set(flat))
    assert set(frequency_columns_for_arm(columns, "exact_only")) == {
        "age__freq",
        "gender__freq",
    }
    assert frequency_columns_for_arm(columns, "rounded_only") == [
        "daily_screen_time_hours__round0_freq"
    ]
