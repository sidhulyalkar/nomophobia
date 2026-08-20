from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .config import CAT_COLS, DECIMAL_COLS, RAW_COLS

_MISSING_INT = np.int64(-9_000_000_000_000_000_000)
_MISSING_CAT = "__missing__"
_SEVERITY_ORDER = {"none": 0.0, "mild": 1.0, "moderate": 2.0, "severe": 3.0}


def _canonical_token(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def normalize_source_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Normalize common source-dataset column spellings to the competition schema."""
    aliases = {
        _canonical_token(column): column
        for column in RAW_COLS + ["addicted_label", "addiction_level"]
    }
    rename = {}
    for column in source.columns:
        token = _canonical_token(column)
        if token in aliases:
            rename[column] = aliases[token]
    out = source.rename(columns=rename).copy()
    missing = [column for column in RAW_COLS if column not in out.columns]
    if missing:
        raise ValueError(f"source dataset is missing required columns: {missing}")
    return out


def _binary_source_label(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        y = numeric.to_numpy(float)
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("source addicted_label must be binary")
        return y.astype(np.int8)
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "notaddicted": 0,
        "not addicted": 0,
        "addicted": 1,
    }
    normalized = values.astype("string").str.strip().str.lower()
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        bad = sorted(normalized[mapped.isna()].dropna().unique().tolist())[:10]
        raise ValueError(f"unrecognized source addicted_label values: {bad}")
    return mapped.to_numpy(np.int8)


def _source_severity(values: pd.Series) -> np.ndarray:
    normalized = values.astype("string").str.strip().str.lower()
    mapped = normalized.map(_SEVERITY_ORDER)
    if mapped.isna().any():
        bad = sorted(normalized[mapped.isna()].dropna().unique().tolist())[:10]
        raise ValueError(f"unrecognized addiction_level values: {bad}")
    return mapped.to_numpy(float)


def _normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in RAW_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    out: dict[str, object] = {}
    for column in RAW_COLS:
        if column in CAT_COLS:
            out[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.lower()
                .fillna(_MISSING_CAT)
            )
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
        scale = 100.0 if column in DECIMAL_COLS else 1.0
        rounded = np.rint(values * scale)
        encoded = np.full(len(df), _MISSING_INT, dtype=np.int64)
        finite = np.isfinite(rounded)
        encoded[finite] = rounded[finite].astype(np.int64)
        out[column] = encoded
    return pd.DataFrame(out, index=df.index)


def _hash_group(frame: pd.DataFrame, group: tuple[str, ...]) -> np.ndarray:
    return pd.util.hash_pandas_object(
        frame.loc[:, list(group)], index=False
    ).to_numpy(np.uint64)


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))


def _logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


@dataclass(frozen=True)
class SourceGroupStats:
    group: tuple[str, ...]
    coverage: float
    source_unique: int
    source_count: pd.Series
    source_positive_count: pd.Series | None
    source_severity_sum: pd.Series | None


@dataclass(frozen=True)
class SourceLineageEncoder:
    groups: tuple[SourceGroupStats, ...]
    source_positive_prior: float | None
    source_severity_prior: float | None
    max_order: int
    screen_rows: int
    selection_report: tuple[dict, ...]

    @property
    def feature_group_count(self) -> int:
        return len(self.groups)


def _group_candidates(max_order: int) -> list[tuple[str, ...]]:
    if max_order < 1 or max_order > len(RAW_COLS):
        raise ValueError("max_order must be between 1 and the number of raw columns")
    return [
        group
        for order in range(1, max_order + 1)
        for group in combinations(RAW_COLS, order)
    ]


def _sample_reference(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    if rows <= 0 or rows >= len(frame):
        return frame
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(frame), size=rows, replace=False))
    return frame.iloc[idx]


def fit_source_lineage_encoder(
    source: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    max_order: int = 3,
    max_groups: int = 96,
    screen_rows: int = 60_000,
    seed: int = 20260819,
    min_coverage: float = 5e-4,
    max_coverage: float = 0.9995,
) -> SourceLineageEncoder:
    """Fit exact source-subset signatures without reading competition labels.

    Candidate subsets are ranked only by the entropy of their exact membership
    indicator on a target-free train+test reference sample. Source labels and
    source severity are retained for an optional local-support treatment, but
    never participate in group selection.
    """
    if max_groups <= 0:
        raise ValueError("max_groups must be positive")
    src = normalize_source_columns(source).reset_index(drop=True)
    ref = reference.loc[:, RAW_COLS].reset_index(drop=True)
    src_norm = _normalize_values(src)
    ref_norm = _normalize_values(ref)
    ref_sample = _sample_reference(ref_norm, screen_rows, seed)

    candidates = []
    for group in _group_candidates(max_order):
        source_hash = _hash_group(src_norm, group)
        source_unique = np.unique(source_hash)
        probe_hash = _hash_group(ref_sample, group)
        coverage = float(
            np.isin(probe_hash, source_unique, assume_unique=False).mean()
        )
        entropy = _binary_entropy(coverage)
        if min_coverage <= coverage <= max_coverage and entropy > 0:
            score = entropy * (1.0 + 0.08 * (len(group) - 1))
            candidates.append((score, coverage, group, source_hash))

    if not candidates:
        raise ValueError(
            "no non-degenerate source-membership groups survived screening"
        )

    by_order: dict[int, list] = {}
    for row in candidates:
        by_order.setdefault(len(row[2]), []).append(row)
    for rows in by_order.values():
        rows.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)

    selected = []
    base_quota = max(1, max_groups // max(1, len(by_order)))
    for order in sorted(by_order):
        selected.extend(by_order[order][:base_quota])
    chosen_groups = {row[2] for row in selected}
    if len(selected) < max_groups:
        leftovers = sorted(
            candidates,
            key=lambda row: (row[0], row[1], row[2]),
            reverse=True,
        )
        for row in leftovers:
            if row[2] in chosen_groups:
                continue
            selected.append(row)
            chosen_groups.add(row[2])
            if len(selected) >= max_groups:
                break
    selected = sorted(
        selected[:max_groups], key=lambda row: (len(row[2]), -row[0], row[2])
    )

    source_y = (
        _binary_source_label(src["addicted_label"])
        if "addicted_label" in src
        else None
    )
    severity = (
        _source_severity(src["addiction_level"])
        if "addiction_level" in src
        else None
    )
    positive_prior = None if source_y is None else float(source_y.mean())
    severity_prior = None if severity is None else float(severity.mean())

    groups = []
    report = []
    for score, coverage, group, source_hash in selected:
        counts = pd.Series(source_hash).value_counts(sort=False)
        positive_counts = None
        severity_sum = None
        if source_y is not None:
            positive_counts = (
                pd.DataFrame({"key": source_hash, "value": source_y})
                .groupby("key", sort=False)["value"]
                .sum()
            )
        if severity is not None:
            severity_sum = (
                pd.DataFrame({"key": source_hash, "value": severity})
                .groupby("key", sort=False)["value"]
                .sum()
            )
        groups.append(
            SourceGroupStats(
                group=group,
                coverage=float(coverage),
                source_unique=int(counts.size),
                source_count=counts,
                source_positive_count=positive_counts,
                source_severity_sum=severity_sum,
            )
        )
        report.append(
            {
                "group": list(group),
                "order": len(group),
                "coverage": float(coverage),
                "membership_entropy": _binary_entropy(float(coverage)),
                "screen_score": float(score),
                "source_unique": int(counts.size),
            }
        )

    return SourceLineageEncoder(
        groups=tuple(groups),
        source_positive_prior=positive_prior,
        source_severity_prior=severity_prior,
        max_order=max_order,
        screen_rows=min(int(screen_rows), len(ref_norm)),
        selection_report=tuple(report),
    )


def add_source_lineage_features(
    df: pd.DataFrame,
    encoder: SourceLineageEncoder,
    *,
    include_source_labels: bool = False,
    source_smoothing: float = 8.0,
) -> pd.DataFrame:
    """Append source-membership and optional discrete local-source support."""
    if source_smoothing <= 0:
        raise ValueError("source_smoothing must be positive")
    frame = df.loc[:, RAW_COLS].copy()
    normalized = _normalize_values(frame)
    additions: dict[str, object] = {}
    seen_total = np.zeros(len(frame), dtype=np.int16)
    weighted_seen = np.zeros(len(frame), dtype=np.int16)
    log_count_sum = np.zeros(len(frame), dtype=np.float32)
    posterior_sum = np.zeros(len(frame), dtype=np.float32)
    posterior_seen = np.zeros(len(frame), dtype=np.int16)
    severity_sum_centered = np.zeros(len(frame), dtype=np.float32)

    if include_source_labels and encoder.source_positive_prior is None:
        raise ValueError(
            "source labels requested but encoder was fit without addicted_label"
        )

    for stats in encoder.groups:
        group_name = "__".join(stats.group)
        keys = _hash_group(normalized, stats.group)
        counts = (
            pd.Series(keys)
            .map(stats.source_count)
            .fillna(0)
            .to_numpy(np.float32)
        )
        seen = counts > 0
        seen_u8 = seen.astype(np.uint8)
        log_count = np.log1p(counts).astype(np.float32)
        additions[f"srcmem__{group_name}"] = seen_u8
        additions[f"srccount__{group_name}"] = log_count
        seen_total += seen_u8
        weighted_seen += seen_u8.astype(np.int16) * len(stats.group)
        log_count_sum += log_count

        if include_source_labels:
            assert encoder.source_positive_prior is not None
            assert stats.source_positive_count is not None
            prior = float(encoder.source_positive_prior)
            positives = (
                pd.Series(keys)
                .map(stats.source_positive_count)
                .fillna(0)
                .to_numpy(np.float32)
            )
            posterior = (positives + source_smoothing * prior) / (
                counts + source_smoothing
            )
            centered = (_logit(posterior) - _logit(prior)).astype(np.float32)
            centered[~seen] = 0.0
            additions[f"srcodds__{group_name}"] = centered
            posterior_sum += centered
            posterior_seen += seen_u8

            if (
                encoder.source_severity_prior is not None
                and stats.source_severity_sum is not None
            ):
                severity_prior = float(encoder.source_severity_prior)
                sums = (
                    pd.Series(keys)
                    .map(stats.source_severity_sum)
                    .fillna(0)
                    .to_numpy(np.float32)
                )
                severity_post = (
                    (sums + source_smoothing * severity_prior)
                    / (counts + source_smoothing)
                    - severity_prior
                ).astype(np.float32)
                severity_post[~seen] = 0.0
                additions[f"srcseverity__{group_name}"] = severity_post
                severity_sum_centered += severity_post

    denom = max(1, len(encoder.groups))
    additions["srcagg__seen_groups"] = seen_total
    additions["srcagg__seen_fraction"] = (seen_total / denom).astype(np.float32)
    additions["srcagg__order_weighted_seen"] = weighted_seen
    additions["srcagg__log_count_sum"] = log_count_sum
    if include_source_labels:
        safe = np.maximum(posterior_seen, 1)
        additions["srcagg__mean_centered_logodds"] = (
            posterior_sum / safe
        ).astype(np.float32)
        additions["srcagg__mean_centered_severity"] = (
            severity_sum_centered / safe
        ).astype(np.float32)

    return pd.concat(
        [frame, pd.DataFrame(additions, index=frame.index)], axis=1
    )


def encoder_report(encoder: SourceLineageEncoder) -> dict:
    return {
        "feature_group_count": encoder.feature_group_count,
        "max_order": encoder.max_order,
        "screen_rows": encoder.screen_rows,
        "source_positive_prior": encoder.source_positive_prior,
        "source_severity_prior": encoder.source_severity_prior,
        "selected_groups": list(encoder.selection_report),
    }
