from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import CORE_USAGE, RAW_COLS

_NUMERIC_SENTINEL = -9.87654321e15
_STRING_SENTINEL = "__MISSING__"


@dataclass(frozen=True)
class JointDensitySpec:
    name: str
    columns: tuple[str, ...]
    decimals: tuple[int | None, ...]


DEFAULT_PAIR_SPECS = (
    JointDensitySpec("daily_social", ("daily_screen_time_hours", "social_media_hours"), (1, 1)),
    JointDensitySpec("daily_weekend", ("daily_screen_time_hours", "weekend_screen_time"), (1, 1)),
    JointDensitySpec("social_gaming", ("social_media_hours", "gaming_hours"), (1, 1)),
    JointDensitySpec("daily_work", ("daily_screen_time_hours", "work_study_hours"), (1, 1)),
    JointDensitySpec("daily_sleep", ("daily_screen_time_hours", "sleep_hours"), (1, 1)),
    JointDensitySpec("notifications_opens", ("notifications_per_day", "app_opens_per_day"), (0, 0)),
    JointDensitySpec("age_daily", ("age", "daily_screen_time_hours"), (0, 1)),
)

DEFAULT_TRIPLE_SPECS = (
    JointDensitySpec("daily_social_weekend", ("daily_screen_time_hours", "social_media_hours", "weekend_screen_time"), (1, 1, 1)),
    JointDensitySpec("daily_social_gaming", ("daily_screen_time_hours", "social_media_hours", "gaming_hours"), (1, 1, 1)),
)

DEFAULT_REGIME_COLUMNS = tuple(CORE_USAGE)


def _canonical(series: pd.Series, decimals: int | None) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if decimals is not None:
            values = values.round(decimals)
        return values.fillna(_NUMERIC_SENTINEL)
    return series.astype("string").fillna(_STRING_SENTINEL)


def _multi_counts(arrays: Sequence[pd.Series]) -> pd.Series:
    frame = pd.DataFrame({f"k{i}": s.reset_index(drop=True) for i, s in enumerate(arrays)})
    return frame.value_counts(dropna=False, sort=False)


def _lookup_multi(counts: pd.Series, arrays: Sequence[pd.Series]) -> np.ndarray:
    idx = pd.MultiIndex.from_arrays([s.to_numpy() for s in arrays], names=list(counts.index.names))
    return counts.reindex(idx, fill_value=0).to_numpy(dtype=float)


def _marginal_count(reference: pd.Series, values: pd.Series) -> np.ndarray:
    vc = reference.value_counts(dropna=False, sort=False)
    return values.map(vc).fillna(0).to_numpy(dtype=float)


def _safe_log(values: np.ndarray, alpha: float) -> np.ndarray:
    return np.log(np.asarray(values, dtype=float) + float(alpha))


def add_joint_density_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    reference_train: pd.DataFrame,
    reference_test: pd.DataFrame,
    pair_specs: Iterable[JointDensitySpec] = DEFAULT_PAIR_SPECS,
    triple_specs: Iterable[JointDensitySpec] = DEFAULT_TRIPLE_SPECS,
    alpha: float = 1.0,
    include_conditional: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Add target-free pair/triple population-density geometry."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    tr = pd.DataFrame(index=train.index)
    te = pd.DataFrame(index=test.index)
    reference = pd.concat([reference_train.reset_index(drop=True), reference_test.reset_index(drop=True)], ignore_index=True)
    n_reference = float(len(reference))
    metadata = {"pairs": [], "triples": [], "alpha": float(alpha), "reference_rows": int(n_reference)}

    for spec in tuple(pair_specs) + tuple(triple_specs):
        if len(spec.columns) != len(spec.decimals):
            raise ValueError(f"bad density spec {spec.name}: columns/decimals mismatch")
        missing = [c for c in spec.columns if c not in reference.columns]
        if missing:
            raise ValueError(f"density spec {spec.name} missing columns: {missing}")

        ref_keys = [_canonical(reference[c], dec) for c, dec in zip(spec.columns, spec.decimals)]
        train_keys = [_canonical(train[c], dec) for c, dec in zip(spec.columns, spec.decimals)]
        test_keys = [_canonical(test[c], dec) for c, dec in zip(spec.columns, spec.decimals)]
        joint_counts = _multi_counts(ref_keys)
        joint_tr = _lookup_multi(joint_counts, train_keys)
        joint_te = _lookup_multi(joint_counts, test_keys)
        marg_tr = [_marginal_count(r, x) for r, x in zip(ref_keys, train_keys)]
        marg_te = [_marginal_count(r, x) for r, x in zip(ref_keys, test_keys)]

        prefix = f"density__{spec.name}"
        tr[f"{prefix}__freq"] = joint_tr.astype(np.float32)
        te[f"{prefix}__freq"] = joint_te.astype(np.float32)
        tr[f"{prefix}__logfreq"] = np.log1p(joint_tr).astype(np.float32)
        te[f"{prefix}__logfreq"] = np.log1p(joint_te).astype(np.float32)

        order = len(spec.columns)
        log_n = np.log(max(n_reference, 1.0))
        interaction_tr = _safe_log(joint_tr, alpha) + (order - 1) * log_n
        interaction_te = _safe_log(joint_te, alpha) + (order - 1) * log_n
        for marginal in marg_tr:
            interaction_tr -= _safe_log(marginal, alpha)
        for marginal in marg_te:
            interaction_te -= _safe_log(marginal, alpha)
        tr[f"{prefix}__interaction"] = np.clip(interaction_tr, -20, 20).astype(np.float32)
        te[f"{prefix}__interaction"] = np.clip(interaction_te, -20, 20).astype(np.float32)

        if order == 2 and include_conditional:
            tr[f"{prefix}__logp_left_given_right"] = (_safe_log(joint_tr, alpha) - _safe_log(marg_tr[1], alpha)).astype(np.float32)
            te[f"{prefix}__logp_left_given_right"] = (_safe_log(joint_te, alpha) - _safe_log(marg_te[1], alpha)).astype(np.float32)
            tr[f"{prefix}__logp_right_given_left"] = (_safe_log(joint_tr, alpha) - _safe_log(marg_tr[0], alpha)).astype(np.float32)
            te[f"{prefix}__logp_right_given_left"] = (_safe_log(joint_te, alpha) - _safe_log(marg_te[0], alpha)).astype(np.float32)

        record = {"name": spec.name, "columns": list(spec.columns), "decimals": list(spec.decimals), "unique_joint_states": int(len(joint_counts))}
        (metadata["pairs"] if order == 2 else metadata["triples"]).append(record)

    return tr, te, metadata


def _missing_mask_excluding(df: pd.DataFrame, excluded: str) -> np.ndarray:
    mask = np.zeros(len(df), dtype=np.uint16)
    bit = 0
    for column in RAW_COLS:
        if column == excluded:
            continue
        mask |= df[column].isna().to_numpy(dtype=np.uint16) << bit
        bit += 1
    return mask


def _default_decimals(column: str) -> int | None:
    if column in {"daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours", "weekend_screen_time"}:
        return 1
    if column in {"age", "notifications_per_day", "app_opens_per_day"}:
        return 0
    return None


def add_regime_conditioned_density_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    reference_train: pd.DataFrame,
    reference_test: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_REGIME_COLUMNS,
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Count values inside the row's other-column missingness regime."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    reference = pd.concat([reference_train.reset_index(drop=True), reference_test.reset_index(drop=True)], ignore_index=True)
    tr = pd.DataFrame(index=train.index)
    te = pd.DataFrame(index=test.index)
    n_reference = float(len(reference))
    records = []

    for column in columns:
        if column not in reference.columns:
            raise ValueError(f"unknown regime-density column: {column}")
        decimals = _default_decimals(column)
        rv = _canonical(reference[column], decimals)
        tv = _canonical(train[column], decimals)
        qv = _canonical(test[column], decimals)
        rm = pd.Series(_missing_mask_excluding(reference, column))
        tm = pd.Series(_missing_mask_excluding(train, column), index=train.index)
        qm = pd.Series(_missing_mask_excluding(test, column), index=test.index)

        joint = _multi_counts([rv, rm])
        joint_tr = _lookup_multi(joint, [tv, tm])
        joint_te = _lookup_multi(joint, [qv, qm])
        value_tr = _marginal_count(rv, tv)
        value_te = _marginal_count(rv, qv)
        mask_tr = _marginal_count(rm, tm)
        mask_te = _marginal_count(rm, qm)

        prefix = f"density__{column}__regime"
        tr[f"{prefix}_freq"] = joint_tr.astype(np.float32)
        te[f"{prefix}_freq"] = joint_te.astype(np.float32)
        tr[f"{prefix}_logfreq"] = np.log1p(joint_tr).astype(np.float32)
        te[f"{prefix}_logfreq"] = np.log1p(joint_te).astype(np.float32)
        lift_tr = _safe_log(joint_tr, alpha) + np.log(max(n_reference, 1.0)) - _safe_log(value_tr, alpha) - _safe_log(mask_tr, alpha)
        lift_te = _safe_log(joint_te, alpha) + np.log(max(n_reference, 1.0)) - _safe_log(value_te, alpha) - _safe_log(mask_te, alpha)
        tr[f"{prefix}_interaction"] = np.clip(lift_tr, -20, 20).astype(np.float32)
        te[f"{prefix}_interaction"] = np.clip(lift_te, -20, 20).astype(np.float32)
        records.append({"column": column, "decimals": decimals, "unique_value_regime_states": int(len(joint))})

    return tr, te, {"columns": records, "alpha": float(alpha), "reference_rows": int(n_reference)}


def add_source_stability_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    reference_train: pd.DataFrame,
    reference_test: pd.DataFrame,
    columns: Iterable[str] = RAW_COLS,
    alpha: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Measure unsigned train/test support agreement for each marginal value."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    tr = pd.DataFrame(index=train.index)
    te = pd.DataFrame(index=test.index)
    ntr = float(len(reference_train))
    nte = float(len(reference_test))
    records = []

    for column in columns:
        decimals = _default_decimals(column)
        rt = _canonical(reference_train[column], decimals)
        re = _canonical(reference_test[column], decimals)
        tv = _canonical(train[column], decimals)
        qv = _canonical(test[column], decimals)
        ct = rt.value_counts(dropna=False, sort=False)
        ce = re.value_counts(dropna=False, sort=False)

        def values(key: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            a = key.map(ct).fillna(0).to_numpy(dtype=float)
            b = key.map(ce).fillna(0).to_numpy(dtype=float)
            pa = (a + alpha) / (ntr + alpha)
            pb = (b + alpha) / (nte + alpha)
            balance = 1.0 - np.abs(pa - pb) / (pa + pb + 1e-12)
            harmonic = 2.0 * (a + alpha) * (b + alpha) / (a + b + 2 * alpha)
            abs_log_ratio = np.abs(np.log(pa) - np.log(pb))
            min_support = np.minimum(a, b)
            return balance, harmonic, abs_log_ratio, min_support

        atr = values(tv)
        ate = values(qv)
        prefix = f"density__{column}__source_stability"
        for suffix, a, b in zip(("balance", "harmonic_support", "abs_log_ratio", "min_support"), atr, ate):
            tr[f"{prefix}_{suffix}"] = np.asarray(a, dtype=np.float32)
            te[f"{prefix}_{suffix}"] = np.asarray(b, dtype=np.float32)
        records.append({"column": column, "decimals": decimals})

    return tr, te, {"columns": records, "alpha": float(alpha), "note": "unsigned train/test support stability only; no signed source posterior"}
