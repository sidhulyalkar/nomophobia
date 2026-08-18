from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import CAT_COLS, NUM_COLS, RAW_COLS

IDENTITY_SENTINEL = "__MISSING__"
SCREEN_RELATION_FEATURES = [
    "screen__other_hours",
    "screen__weekend_minus_daily",
    "screen__social_minus_gaming",
    "screen__social_minus_work",
    "screen__gaming_minus_work",
    "screen__components_over_daily",
    "screen__leisure_over_daily",
    "screen__weekend_over_daily",
]


def _safe_ratio(a: pd.Series, b: pd.Series, eps: float = 1e-3) -> pd.Series:
    return a / (b.abs() + eps)


def roundtrip_float_key(series: pd.Series) -> pd.Series:
    """Deterministic exact binary64 category key using Python's round-trip hex form."""
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    out = np.empty(len(values), dtype=object)
    missing = np.isnan(values)
    out[missing] = IDENTITY_SENTINEL
    idx = np.where(~missing)[0]
    out[idx] = [float(value).hex() for value in values[idx]]
    return pd.Series(out, index=series.index, dtype="string")


def add_screen_relation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a compact target-free screen-allocation relation block."""
    daily = df["daily_screen_time_hours"]
    social = df["social_media_hours"]
    gaming = df["gaming_hours"]
    work = df["work_study_hours"]
    weekend = df["weekend_screen_time"]
    components = social + gaming + work
    leisure = social + gaming
    extra = pd.DataFrame(
        {
            "screen__other_hours": daily - components,
            "screen__weekend_minus_daily": weekend - daily,
            "screen__social_minus_gaming": social - gaming,
            "screen__social_minus_work": social - work,
            "screen__gaming_minus_work": gaming - work,
            "screen__components_over_daily": _safe_ratio(components, daily),
            "screen__leisure_over_daily": _safe_ratio(leisure, daily),
            "screen__weekend_over_daily": _safe_ratio(weekend, daily),
        },
        index=df.index,
    )
    return pd.concat([df.copy(), extra], axis=1)


def _float64_parts(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    bits = values.view(np.uint64)
    exponent = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.uint16)
    mantissa = bits & np.uint64((1 << 52) - 1)
    low12 = (mantissa & np.uint64(0xFFF)).astype(np.uint16)
    missing = np.isnan(values)
    exponent[missing] = 0
    low12[missing] = 0
    return exponent, low12


def add_identity_digit_features(
    df: pd.DataFrame,
    *,
    columns: Iterable[str] = NUM_COLS,
    include_exact_categories: bool = False,
    round_decimals: tuple[int, ...] = (0, 1, 2, 3),
) -> pd.DataFrame:
    """Add target-free numeric identity, rounding, and binary64 geometry."""
    additions: dict[str, object] = {}
    for column in columns:
        values = df[column].astype("float64")
        additions[f"{column}__is_missing"] = values.isna().astype(np.int8)

        exponent, low12 = _float64_parts(values)
        additions[f"{column}__f64_exponent"] = exponent
        additions[f"{column}__f64_mantissa_low12"] = low12

        for decimals in round_decimals:
            scale = float(10**decimals)
            rounded = values.round(decimals)
            additions[f"{column}__round{decimals}"] = rounded.astype(np.float32)
            additions[f"{column}__round{decimals}_residual"] = (
                values - rounded
            ).astype(np.float32)
            if decimals > 0:
                scaled = np.rint(values * scale)
                digit = np.mod(np.abs(scaled), 10)
                additions[f"{column}__digit{decimals}"] = (
                    pd.Series(digit, index=df.index).fillna(-1).astype(np.int8)
                )

        if include_exact_categories:
            additions[f"{column}__f64_key"] = roundtrip_float_key(values)

    return pd.concat([df.copy(), pd.DataFrame(additions, index=df.index)], axis=1)


def build_contrast_feature_frame(
    df: pd.DataFrame,
    feature_set: str,
    *,
    include_exact_categories: bool = False,
) -> pd.DataFrame:
    """Build a predeclared matched-ablation feature view."""
    missing = [column for column in RAW_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"missing required raw columns: {missing}")
    frame = df[RAW_COLS].copy()
    if feature_set == "raw":
        return frame
    if feature_set == "identity":
        return add_identity_digit_features(
            frame, include_exact_categories=include_exact_categories
        )
    if feature_set == "screen":
        return add_screen_relation_features(frame)
    if feature_set == "identity_screen":
        return add_screen_relation_features(
            add_identity_digit_features(
                frame, include_exact_categories=include_exact_categories
            )
        )
    raise ValueError(
        "feature_set must be one of raw, identity, screen, identity_screen"
    )


def categorical_feature_names(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column in CAT_COLS
        or str(df[column].dtype).startswith(("string", "category"))
        or df[column].dtype == object
    ]
