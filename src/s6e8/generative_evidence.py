from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CAT_COLS, RAW_COLS

_MISSING_CAT = "__missing__"


@dataclass(frozen=True)
class FeatureBinner:
    name: str
    kind: str
    edges: tuple[float, ...] = ()
    categories: tuple[str, ...] = ()

    @property
    def n_states(self) -> int:
        if self.kind == "numeric":
            # finite-value bins plus one explicit missing state
            return len(self.edges) + 2
        # known categories plus one explicit unseen-category state
        return len(self.categories) + 1

    def transform(self, values: pd.Series) -> np.ndarray:
        if self.kind == "numeric":
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
            out = np.full(len(numeric), len(self.edges) + 1, dtype=np.int16)
            finite = np.isfinite(numeric)
            if finite.any():
                out[finite] = np.searchsorted(
                    np.asarray(self.edges, dtype=float), numeric[finite], side="right"
                ).astype(np.int16)
            return out

        normalized = (
            values.astype("string")
            .str.strip()
            .str.lower()
            .fillna(_MISSING_CAT)
        )
        mapping = {value: idx for idx, value in enumerate(self.categories)}
        unseen = len(self.categories)
        return normalized.map(mapping).fillna(unseen).to_numpy(np.int16)


@dataclass(frozen=True)
class TANModel:
    columns: tuple[str, ...]
    binners: tuple[FeatureBinner, ...]
    root: int
    parents: tuple[int, ...]
    marginal_llr: tuple[np.ndarray, ...]
    conditional_llr: tuple[np.ndarray | None, ...]
    prior_logodds: float
    conditional_mutual_information: np.ndarray
    feature_target_mutual_information: tuple[float, ...]
    alpha: float

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        rows = []
        for child, parent in enumerate(self.parents):
            if parent >= 0:
                rows.append((self.columns[parent], self.columns[child]))
        return tuple(rows)


def _fit_binner(name: str, values: pd.Series, n_bins: int) -> FeatureBinner:
    if name in CAT_COLS:
        normalized = (
            values.astype("string")
            .str.strip()
            .str.lower()
            .fillna(_MISSING_CAT)
        )
        categories = tuple(sorted(normalized.unique().tolist()))
        return FeatureBinner(name=name, kind="categorical", categories=categories)

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    if len(finite) == 0:
        return FeatureBinner(name=name, kind="numeric", edges=())
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    edges = np.unique(np.quantile(finite, quantiles)).astype(float)
    return FeatureBinner(name=name, kind="numeric", edges=tuple(edges.tolist()))


def fit_binners(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] | list[str] = tuple(RAW_COLS),
    n_bins: int = 24,
) -> tuple[FeatureBinner, ...]:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    columns = tuple(columns)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return tuple(_fit_binner(name, frame[name], n_bins) for name in columns)


def transform_binned(
    frame: pd.DataFrame,
    binners: tuple[FeatureBinner, ...],
) -> np.ndarray:
    if not binners:
        raise ValueError("at least one binner is required")
    return np.column_stack(
        [binner.transform(frame[binner.name]) for binner in binners]
    ).astype(np.int16, copy=False)


def _mutual_information_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    pxy = counts / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))


def _feature_target_mi(
    codes: np.ndarray,
    y: np.ndarray,
    n_states: int,
) -> float:
    flat = codes.astype(np.int64) * 2 + y.astype(np.int64)
    counts = np.bincount(flat, minlength=n_states * 2).reshape(n_states, 2)
    return _mutual_information_from_counts(counts)


def _conditional_pair_mi(
    left: np.ndarray,
    right: np.ndarray,
    y: np.ndarray,
    left_states: int,
    right_states: int,
) -> float:
    total = len(y)
    value = 0.0
    for cls in (0, 1):
        mask = y == cls
        n_cls = int(mask.sum())
        if n_cls == 0:
            continue
        flat = left[mask].astype(np.int64) * right_states + right[mask].astype(
            np.int64
        )
        counts = np.bincount(
            flat, minlength=left_states * right_states
        ).reshape(left_states, right_states)
        value += (n_cls / total) * _mutual_information_from_counts(counts)
    return float(value)


def _maximum_spanning_tree(weights: np.ndarray, root: int) -> tuple[int, ...]:
    weights = np.asarray(weights, dtype=float)
    n = weights.shape[0]
    if weights.shape != (n, n):
        raise ValueError("weights must be square")
    if not (0 <= root < n):
        raise ValueError("invalid root")
    parents = np.full(n, -2, dtype=np.int16)
    parents[root] = -1
    selected = np.zeros(n, dtype=bool)
    selected[root] = True

    while int(selected.sum()) < n:
        best_weight = -np.inf
        best_parent = -1
        best_child = -1
        for parent in np.flatnonzero(selected):
            for child in np.flatnonzero(~selected):
                weight = float(weights[parent, child])
                if (
                    weight > best_weight
                    or (
                        weight == best_weight
                        and (int(parent), int(child)) < (best_parent, best_child)
                    )
                ):
                    best_weight = weight
                    best_parent = int(parent)
                    best_child = int(child)
        if best_child < 0:
            raise RuntimeError("failed to construct spanning tree")
        parents[best_child] = best_parent
        selected[best_child] = True
    return tuple(int(value) for value in parents)


def _marginal_log_ratio(
    codes: np.ndarray,
    y: np.ndarray,
    n_states: int,
    alpha: float,
) -> np.ndarray:
    rows = []
    for cls in (0, 1):
        mask = y == cls
        counts = np.bincount(codes[mask], minlength=n_states).astype(float)
        probs = (counts + alpha) / (float(mask.sum()) + alpha * n_states)
        rows.append(np.log(probs))
    return rows[1] - rows[0]


def _conditional_log_ratio(
    parent: np.ndarray,
    child: np.ndarray,
    y: np.ndarray,
    parent_states: int,
    child_states: int,
    alpha: float,
) -> np.ndarray:
    log_probs = []
    for cls in (0, 1):
        mask = y == cls
        flat = parent[mask].astype(np.int64) * child_states + child[mask].astype(
            np.int64
        )
        counts = np.bincount(
            flat, minlength=parent_states * child_states
        ).reshape(parent_states, child_states).astype(float)
        parent_counts = counts.sum(axis=1, keepdims=True)
        probs = (counts + alpha) / (parent_counts + alpha * child_states)
        log_probs.append(np.log(probs))
    return log_probs[1] - log_probs[0]


def fit_tan_model(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    columns: tuple[str, ...] | list[str] = tuple(RAW_COLS),
    n_bins: int = 24,
    alpha: float = 1.0,
) -> TANModel:
    """Fit a Tree-Augmented Naive Bayes likelihood-ratio model.

    Numeric discretization is target-free.  The TAN dependency tree is learned
    only from the supplied training rows by maximizing conditional mutual
    information I(X_i; X_j | Y), so ordinary outer-fold use is leakage-safe.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    columns = tuple(columns)
    y = np.asarray(y, dtype=np.int8)
    if len(frame) != len(y):
        raise ValueError("frame and y must align")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("y must contain both binary classes")

    binners = fit_binners(frame, columns=columns, n_bins=n_bins)
    codes = transform_binned(frame, binners)
    n_features = len(columns)
    n_states = [binner.n_states for binner in binners]

    target_mi = tuple(
        _feature_target_mi(codes[:, idx], y, n_states[idx])
        for idx in range(n_features)
    )
    root = int(np.argmax(np.asarray(target_mi)))

    cmi = np.zeros((n_features, n_features), dtype=float)
    for left in range(n_features):
        for right in range(left + 1, n_features):
            value = _conditional_pair_mi(
                codes[:, left],
                codes[:, right],
                y,
                n_states[left],
                n_states[right],
            )
            cmi[left, right] = value
            cmi[right, left] = value
    parents = _maximum_spanning_tree(cmi, root)

    marginals = tuple(
        _marginal_log_ratio(codes[:, idx], y, n_states[idx], alpha)
        for idx in range(n_features)
    )
    conditionals: list[np.ndarray | None] = [None] * n_features
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        conditionals[child] = _conditional_log_ratio(
            codes[:, parent],
            codes[:, child],
            y,
            n_states[parent],
            n_states[child],
            alpha,
        )

    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    prior_logodds = float(np.log((n1 + alpha) / (n0 + alpha)))
    return TANModel(
        columns=columns,
        binners=binners,
        root=root,
        parents=parents,
        marginal_llr=marginals,
        conditional_llr=tuple(conditionals),
        prior_logodds=prior_logodds,
        conditional_mutual_information=cmi,
        feature_target_mutual_information=target_mi,
        alpha=float(alpha),
    )


def score_tan_model(
    model: TANModel,
    frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    codes = transform_binned(frame, model.binners)
    n = len(frame)
    naive = np.full(n, model.prior_logodds, dtype=float)
    for idx, llr in enumerate(model.marginal_llr):
        naive += llr[codes[:, idx]]

    tan = np.full(n, model.prior_logodds, dtype=float)
    root = model.root
    tan += model.marginal_llr[root][codes[:, root]]
    for child, parent in enumerate(model.parents):
        if parent < 0:
            continue
        table = model.conditional_llr[child]
        assert table is not None
        tan += table[codes[:, parent], codes[:, child]]
    return {
        "naive": naive,
        "tan": tan,
        "dependency": tan - naive,
    }


def tan_report(model: TANModel) -> dict:
    edge_rows = []
    for child, parent in enumerate(model.parents):
        if parent < 0:
            continue
        edge_rows.append(
            {
                "parent": model.columns[parent],
                "child": model.columns[child],
                "conditional_mutual_information": float(
                    model.conditional_mutual_information[parent, child]
                ),
            }
        )
    edge_rows.sort(
        key=lambda row: row["conditional_mutual_information"], reverse=True
    )
    return {
        "root": model.columns[model.root],
        "alpha": model.alpha,
        "feature_target_mutual_information": {
            name: float(value)
            for name, value in zip(
                model.columns, model.feature_target_mutual_information
            )
        },
        "edges": edge_rows,
    }
