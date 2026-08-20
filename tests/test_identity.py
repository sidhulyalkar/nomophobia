import numpy as np
import pandas as pd

from s6e8.identity import (
    SCREEN_RELATION_FEATURES,
    add_identity_digit_features,
    add_screen_relation_features,
    build_contrast_feature_frame,
    roundtrip_float_key,
)


def _frame():
    return pd.DataFrame(
        {
            "age": [18.0, 19.0, np.nan],
            "daily_screen_time_hours": [7.1, 8.2, 6.3],
            "social_media_hours": [3.1, 2.2, 1.3],
            "gaming_hours": [1.0, 2.0, 1.5],
            "work_study_hours": [2.0, 2.5, 2.2],
            "sleep_hours": [7.0, 6.5, 8.0],
            "notifications_per_day": [120.0, 90.0, 70.0],
            "app_opens_per_day": [80.0, 60.0, 50.0],
            "weekend_screen_time": [9.0, 10.0, 7.0],
            "gender": ["Male", "Female", None],
            "stress_level": ["Medium", "High", "Low"],
            "academic_work_impact": ["Yes", "No", "Yes"],
        }
    )


def test_roundtrip_key_distinguishes_adjacent_float64_values():
    first = np.float64(0.1)
    second = np.nextafter(first, np.float64(1.0))
    keys = roundtrip_float_key(pd.Series([first, second, np.nan]))
    assert keys.iloc[0] != keys.iloc[1]
    assert keys.iloc[2] == "__MISSING__"


def test_identity_features_preserve_rows_and_add_representation_block():
    frame = _frame()
    enriched = add_identity_digit_features(frame, include_exact_categories=True)
    assert len(enriched) == len(frame)
    assert enriched.shape[1] > frame.shape[1] + 30
    assert "daily_screen_time_hours__f64_key" in enriched
    assert "daily_screen_time_hours__round2_residual" in enriched
    assert np.isfinite(enriched["daily_screen_time_hours__f64_exponent"]).all()


def test_screen_relations_are_complete_and_target_free():
    frame = _frame()
    enriched = add_screen_relation_features(frame)
    assert all(column in enriched for column in SCREEN_RELATION_FEATURES)
    expected = (
        frame["daily_screen_time_hours"]
        - frame["social_media_hours"]
        - frame["gaming_hours"]
        - frame["work_study_hours"]
    )
    assert np.allclose(enriched["screen__other_hours"], expected, equal_nan=True)


def test_contrast_feature_sets_are_nested():
    frame = _frame()
    raw = build_contrast_feature_frame(frame, "raw")
    screen = build_contrast_feature_frame(frame, "screen")
    identity = build_contrast_feature_frame(frame, "identity")
    both = build_contrast_feature_frame(frame, "identity_screen")
    assert set(raw.columns).issubset(screen.columns)
    assert set(raw.columns).issubset(identity.columns)
    assert set(screen.columns).issubset(both.columns)
    assert set(identity.columns).issubset(both.columns)
