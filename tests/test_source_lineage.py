from __future__ import annotations

import numpy as np
import pandas as pd

from s6e8.config import RAW_COLS
from s6e8.source_lineage import (
    add_source_lineage_features,
    encoder_report,
    fit_source_lineage_encoder,
    normalize_source_columns,
)


def _source_fixture(n: int = 160) -> pd.DataFrame:
    rows = []
    levels = ["None", "Mild", "Moderate", "Severe"]
    for idx in range(n):
        level = levels[idx % 4]
        rows.append(
            {
                "age": 18 + idx % 18,
                "daily_screen_time_hours": 3.0 + (idx % 50) / 10,
                "social_media_hours": 0.8 + (idx % 24) / 10,
                "gaming_hours": 0.2 + (idx % 17) / 10,
                "work_study_hours": 1.0 + (idx % 30) / 10,
                "sleep_hours": 5.0 + (idx % 28) / 10,
                "notifications_per_day": 20 + idx % 110,
                "app_opens_per_day": 10 + idx % 90,
                "weekend_screen_time": 4.0 + (idx % 60) / 10,
                "gender": ["Male", "Female", "Other"][idx % 3],
                "stress_level": ["Low", "Medium", "High"][idx % 3],
                "academic_work_impact": ["No", "Yes"][idx % 2],
                "addiction_level": level,
                "addicted_label": int(level in {"Moderate", "Severe"}),
            }
        )
    return pd.DataFrame(rows)


def _reference_fixture(source: pd.DataFrame) -> pd.DataFrame:
    reference = pd.concat(
        [source[RAW_COLS], source[RAW_COLS]], ignore_index=True
    ).copy()
    mutate = np.arange(len(reference)) % 3 == 0
    reference.loc[mutate, "daily_screen_time_hours"] += 0.03
    reference.loc[mutate, "notifications_per_day"] += 1000
    reference.loc[np.arange(len(reference)) % 11 == 0, "sleep_hours"] = np.nan
    return reference


def test_source_column_normalization_accepts_public_dataset_style_names():
    source = _source_fixture(20).rename(
        columns={column: "_".join(part.title() for part in column.split("_")) for column in RAW_COLS}
    )
    source = source.rename(
        columns={"addiction_level": "Addiction_Level", "addicted_label": "Addicted_Label"}
    )
    normalized = normalize_source_columns(source)
    assert set(RAW_COLS).issubset(normalized.columns)
    assert "addiction_level" in normalized
    assert "addicted_label" in normalized


def test_group_selection_is_target_free_with_respect_to_competition_labels():
    source = _source_fixture()
    reference = _reference_fixture(source)
    with_target_a = reference.copy()
    with_target_b = reference.copy()
    with_target_a["addicted_label"] = np.arange(len(reference)) % 2
    with_target_b["addicted_label"] = 1 - with_target_a["addicted_label"]

    encoder_a = fit_source_lineage_encoder(
        source, with_target_a, max_order=2, max_groups=12, screen_rows=200, seed=7
    )
    encoder_b = fit_source_lineage_encoder(
        source, with_target_b, max_order=2, max_groups=12, screen_rows=200, seed=7
    )
    assert encoder_report(encoder_a)["selected_groups"] == encoder_report(encoder_b)[
        "selected_groups"
    ]


def test_membership_features_do_not_change_when_source_labels_are_flipped():
    source = _source_fixture()
    reference = _reference_fixture(source)
    flipped = source.copy()
    flipped["addicted_label"] = 1 - flipped["addicted_label"]

    encoder_a = fit_source_lineage_encoder(
        source, reference, max_order=2, max_groups=12, screen_rows=200, seed=11
    )
    encoder_b = fit_source_lineage_encoder(
        flipped, reference, max_order=2, max_groups=12, screen_rows=200, seed=11
    )
    features_a = add_source_lineage_features(
        reference, encoder_a, include_source_labels=False
    )
    features_b = add_source_lineage_features(
        reference, encoder_b, include_source_labels=False
    )
    lineage_columns = [
        column
        for column in features_a
        if column.startswith("srcmem__") or column.startswith("srccount__")
    ]
    pd.testing.assert_frame_equal(
        features_a[lineage_columns], features_b[lineage_columns]
    )


def test_label_aware_local_support_is_finite_and_distinct_from_membership_only():
    source = _source_fixture()
    reference = _reference_fixture(source)
    encoder = fit_source_lineage_encoder(
        source, reference, max_order=2, max_groups=12, screen_rows=200, seed=19
    )
    membership = add_source_lineage_features(
        reference, encoder, include_source_labels=False
    )
    lineage = add_source_lineage_features(
        reference, encoder, include_source_labels=True, source_smoothing=4.0
    )
    odds = [column for column in lineage if column.startswith("srcodds__")]
    severity = [column for column in lineage if column.startswith("srcseverity__")]
    assert odds
    assert severity
    assert lineage.shape[1] > membership.shape[1]
    assert np.isfinite(lineage[odds + severity].to_numpy(float)).all()
    assert "srcagg__seen_fraction" in lineage
    assert lineage["srcagg__seen_fraction"].between(0.0, 1.0).all()
