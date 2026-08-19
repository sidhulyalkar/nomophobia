from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .contrast import rank01
from .pairwise_auc import PairwiseBasis, fit_pairwise_basis, transform_pairwise_basis


@dataclass(frozen=True)
class OffsetRankNet:
    basis: PairwiseBasis
    coefficients: np.ndarray
    anchor_temperature: float
    l2: float
    pair_total: int
    uniform_pairs: int
    hard_pairs: int
    offsets: tuple[int, ...]
    optimizer_success: bool
    optimizer_iterations: int
    final_loss: float


def _oriented_pairs(
    left: np.ndarray,
    right: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = y[left] != y[right]
    left = left[mask]
    right = right[mask]
    pos = np.where(y[left] == 1, left, right)
    neg = np.where(y[left] == 0, left, right)
    return pos.astype(np.int64, copy=False), neg.astype(np.int64, copy=False)


def _sample_pairs(
    y: np.ndarray,
    anchor: np.ndarray,
    *,
    max_pairs: int,
    uniform_fraction: float,
    offsets: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if not 0.0 < uniform_fraction < 1.0:
        raise ValueError("uniform_fraction must lie inside (0, 1)")
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    n_uniform = int(round(max_pairs * uniform_fraction))
    pos_uniform = rng.choice(positives, size=n_uniform, replace=True)
    neg_uniform = rng.choice(negatives, size=n_uniform, replace=True)

    hard_budget = max_pairs - n_uniform
    per_offset = max(1, hard_budget // max(1, len(offsets)))
    order = np.argsort(anchor, kind="mergesort")
    hard_pos = []
    hard_neg = []
    for offset in offsets:
        if offset <= 0 or offset >= len(order):
            continue
        pos, neg = _oriented_pairs(order[:-offset], order[offset:], y)
        if len(pos) > per_offset:
            take = np.sort(rng.choice(len(pos), size=per_offset, replace=False))
            pos = pos[take]
            neg = neg[take]
        hard_pos.append(pos)
        hard_neg.append(neg)
    if hard_pos:
        pos_hard = np.concatenate(hard_pos)
        neg_hard = np.concatenate(hard_neg)
    else:
        pos_hard = np.empty(0, dtype=np.int64)
        neg_hard = np.empty(0, dtype=np.int64)

    pos = np.concatenate([pos_uniform, pos_hard])
    neg = np.concatenate([neg_uniform, neg_hard])
    # Equal total mass for global AUC pairs and hard-scale pairs, regardless of
    # how many hard pairs each offset happened to yield.
    weights = np.empty(len(pos), dtype=np.float32)
    weights[: len(pos_uniform)] = 0.5 / max(1, len(pos_uniform))
    weights[len(pos_uniform) :] = 0.5 / max(1, len(pos_hard))
    weights *= len(weights)
    return pos, neg, weights, len(pos_uniform), len(pos_hard)


def fit_offset_ranknet(
    frame,
    y: np.ndarray,
    anchor: np.ndarray,
    *,
    max_pairs: int = 240_000,
    uniform_fraction: float = 0.50,
    offsets: tuple[int, ...] = (64, 256, 1024, 4096, 16384, 65536, 131072),
    anchor_temperature: float = 12.0,
    l2: float = 2e-3,
    max_iter: int = 28,
    seed: int = 20260819,
) -> OffsetRankNet:
    """Fit a pairwise residual with the champion margin frozen as an offset.

    For oriented positive/negative pairs the loss is

        softplus(-(T * (anchor_pos-anchor_neg) + beta @ (x_pos-x_neg)))

    so beta can only explain ordering evidence that remains after the champion.
    """
    y = np.asarray(y, dtype=np.int8)
    anchor = np.asarray(anchor, dtype=float)
    if len(frame) != len(y) or len(y) != len(anchor):
        raise ValueError("frame, y, and anchor must align")
    if len(np.unique(y)) != 2:
        raise ValueError("both binary classes are required")
    if max_pairs < 10_000:
        raise ValueError("max_pairs is too small")
    if anchor_temperature <= 0 or l2 < 0:
        raise ValueError("invalid optimization constants")

    basis = fit_pairwise_basis(frame)
    matrix = transform_pairwise_basis(frame, basis)
    anchor_rank = rank01(anchor)
    pos, neg, weights, n_uniform, n_hard = _sample_pairs(
        y,
        anchor_rank,
        max_pairs=max_pairs,
        uniform_fraction=uniform_fraction,
        offsets=offsets,
        seed=seed,
    )
    diff = (matrix[pos] - matrix[neg]).astype(np.float64, copy=False)
    margin = anchor_temperature * (anchor_rank[pos] - anchor_rank[neg])
    weights64 = weights.astype(np.float64, copy=False)
    weight_sum = float(weights64.sum())

    def objective(beta: np.ndarray):
        z = margin + diff @ beta
        # softplus(-z), written stably.
        loss_vec = np.logaddexp(0.0, -z)
        loss = float(np.dot(weights64, loss_vec) / weight_sum)
        if l2:
            loss += 0.5 * l2 * float(beta @ beta)
        # d softplus(-z) / dz = -sigmoid(-z)
        grad_factor = -1.0 / (1.0 + np.exp(np.clip(z, -50.0, 50.0)))
        grad = diff.T @ (weights64 * grad_factor) / weight_sum
        if l2:
            grad = grad + l2 * beta
        return loss, grad

    initial = np.zeros(matrix.shape[1], dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-10, "gtol": 1e-6},
    )
    beta = np.asarray(result.x, dtype=np.float32)
    if not np.isfinite(beta).all():
        raise RuntimeError("offset RankNet produced non-finite coefficients")
    return OffsetRankNet(
        basis=basis,
        coefficients=beta,
        anchor_temperature=float(anchor_temperature),
        l2=float(l2),
        pair_total=int(len(pos)),
        uniform_pairs=int(n_uniform),
        hard_pairs=int(n_hard),
        offsets=tuple(int(value) for value in offsets),
        optimizer_success=bool(result.success),
        optimizer_iterations=int(result.nit),
        final_loss=float(result.fun),
    )


def score_offset_ranknet(
    model: OffsetRankNet,
    frame,
    anchor: np.ndarray,
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=float)
    if len(frame) != len(anchor):
        raise ValueError("frame and anchor must align")
    matrix = transform_pairwise_basis(frame, model.basis)
    residual = matrix @ model.coefficients
    score = model.anchor_temperature * rank01(anchor) + residual
    if not np.isfinite(score).all():
        raise RuntimeError("offset RankNet produced non-finite scores")
    return np.asarray(score, dtype=float)


def offset_ranknet_report(model: OffsetRankNet, *, top_n: int = 20) -> dict:
    coef = np.asarray(model.coefficients, dtype=float)
    order = np.argsort(np.abs(coef))[::-1][:top_n]
    return {
        "feature_count": len(model.basis.feature_names),
        "pair_total": model.pair_total,
        "uniform_pairs": model.uniform_pairs,
        "hard_pairs": model.hard_pairs,
        "offsets": list(model.offsets),
        "anchor_temperature": model.anchor_temperature,
        "l2": model.l2,
        "optimizer_success": model.optimizer_success,
        "optimizer_iterations": model.optimizer_iterations,
        "final_loss": model.final_loss,
        "top_coefficients": [
            {
                "feature": model.basis.feature_names[int(idx)],
                "coefficient": float(coef[int(idx)]),
            }
            for idx in order
        ],
    }
