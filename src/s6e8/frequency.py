from __future__ import annotations

from collections.abc import Iterable

from .config import CAT_COLS, DECIMAL_COLS, NUM_COLS


FREQUENCY_ARMS = (
    "full",
    "none",
    "exact_only",
    "rounded_only",
    "categorical_only",
    "exact_plus_rounded",
)


def frequency_feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Return mutually exclusive frequency subfamilies present in a feature frame."""

    present = set(columns)
    groups = {
        "exact_numeric": [f"{c}__freq" for c in NUM_COLS],
        "log_numeric": [f"{c}__logfreq" for c in NUM_COLS],
        "exact_categorical": [f"{c}__freq" for c in CAT_COLS],
        "log_categorical": [f"{c}__logfreq" for c in CAT_COLS],
        "rounded_numeric": [
            f"{c}__round{dec}_freq" for c in DECIMAL_COLS for dec in (0, 1)
        ],
    }
    return {
        name: [column for column in values if column in present]
        for name, values in groups.items()
    }


def frequency_columns_for_arm(columns: Iterable[str], arm: str) -> list[str]:
    """Select frequency columns for one pre-registered ablation arm.

    Non-frequency features are deliberately not returned. The experiment runner combines
    this selection with the unchanged non-frequency backbone so every arm differs only in
    the frequency subfamily under test.
    """

    if arm not in FREQUENCY_ARMS:
        raise ValueError(f"Unknown frequency arm {arm!r}; expected one of {FREQUENCY_ARMS}")
    groups = frequency_feature_groups(columns)
    all_frequency = [column for values in groups.values() for column in values]
    if arm == "full":
        return all_frequency
    if arm == "none":
        return []
    if arm == "exact_only":
        return groups["exact_numeric"] + groups["exact_categorical"]
    if arm == "rounded_only":
        return groups["rounded_numeric"]
    if arm == "categorical_only":
        return groups["exact_categorical"] + groups["log_categorical"]
    if arm == "exact_plus_rounded":
        return (
            groups["exact_numeric"]
            + groups["exact_categorical"]
            + groups["rounded_numeric"]
        )
    raise AssertionError(arm)
