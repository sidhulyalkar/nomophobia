import numpy as np

from s6e8.contrast import (
    ResidualGate,
    first_passing_weight,
    orthogonalize_direction,
    orthogonalize_train_test,
    rank_direction,
    rotating_residual_gate,
)


def test_rank_direction_is_aligned_and_centered():
    control = np.array([0.1, 0.4, 0.3, 0.8, 0.2])
    treatment = np.array([0.2, 0.3, 0.5, 0.9, 0.1])
    direction = rank_direction(treatment, control)
    assert direction.shape == control.shape
    assert abs(float(direction.mean())) < 1e-12


def test_orthogonalize_direction_removes_linear_projection():
    x = np.linspace(-1.0, 1.0, 200)
    z = np.sin(np.arange(200) / 11.0)
    direction = 3.0 * x + z
    residual = orthogonalize_direction(direction, x)
    assert abs(float(np.corrcoef(residual, x)[0, 1])) < 1e-10


def test_first_passing_weight_uses_smallest_stable_point():
    rng = np.random.default_rng(7)
    n = 500
    y = np.array([0, 1] * (n // 2), dtype=np.int8)
    folds = np.arange(n) % 5
    ids = rng.permutation(np.arange(1000, 1000 + n))
    anchor = 0.45 * y + rng.normal(0, 0.50, n)
    direction = 0.35 * y + rng.normal(0, 0.10, n)
    gate = ResidualGate(
        min_gain=1e-5,
        min_fold_wins=4,
        fold_tolerance=-1e-4,
        require_id_slices=True,
        slice_tolerance=-1e-4,
    )
    weight, metrics = first_passing_weight(
        y,
        anchor,
        direction,
        folds,
        [0.0, 0.05, 0.10, 0.20, 0.40],
        ids=ids,
        gate=gate,
    )
    assert weight > 0
    assert metrics["gain"] > 0
    assert metrics["fold_wins"] >= 4


def test_rotating_gate_returns_honest_oof_and_deploy_weight():
    rng = np.random.default_rng(11)
    n = 600
    y = np.array([0, 1] * (n // 2), dtype=np.int8)
    folds = np.arange(n) % 5
    ids = np.arange(n)
    anchor = 0.35 * y + rng.normal(0, 0.60, n)
    direction = 0.45 * y + rng.normal(0, 0.10, n)
    result = rotating_residual_gate(
        y,
        anchor,
        direction,
        folds,
        [0.0, 0.05, 0.10, 0.20, 0.40],
        ids=ids,
        gate=ResidualGate(
            min_gain=1e-5,
            min_fold_wins=4,
            fold_tolerance=-1e-3,
            require_id_slices=True,
            slice_tolerance=-1e-3,
        ),
    )
    assert len(result["honest_oof"]) == n
    assert len(result["selected_weights"]) == 5
    assert result["selected_weight_median"] >= 0
    assert (
        result["honest_metrics"]["candidate_auc"]
        >= result["honest_metrics"]["anchor_auc"]
    )


def test_no_direction_returns_zero_weight():
    n = 200
    y = np.array([0, 1] * 100, dtype=np.int8)
    folds = np.arange(n) % 5
    anchor = np.linspace(0.0, 1.0, n)
    weight, _ = first_passing_weight(
        y,
        anchor,
        np.zeros(n),
        folds,
        [0.0, 0.01, 0.02],
        ids=np.arange(n),
        gate=ResidualGate(min_gain=1e-6, min_fold_wins=4),
    )
    assert weight == 0.0


def test_train_test_orthogonalizer_reuses_oof_projection():
    x_oof = np.linspace(-2.0, 2.0, 300)
    x_test = np.linspace(-1.5, 2.5, 80)
    d_oof = 2.5 * x_oof + np.sin(np.arange(300) / 9.0)
    d_test = 2.5 * x_test + np.cos(np.arange(80) / 7.0)
    residual_oof, residual_test = orthogonalize_train_test(
        d_oof, d_test, x_oof, x_test
    )
    assert abs(float(np.corrcoef(residual_oof, x_oof)[0, 1])) < 1e-10
    assert len(residual_test) == len(x_test)
