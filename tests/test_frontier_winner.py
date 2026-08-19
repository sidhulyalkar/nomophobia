from __future__ import annotations

import numpy as np

from s6e8.winner import (
    all_stability_slices_positive,
    compose_rank_score,
    nested_residual_selection,
    stability_diagnostics,
    validate_direction,
)


def _fixture(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = 2500
    latent = rng.normal(size=n)
    y = (latent + 0.25 * rng.normal(size=n) > 0).astype(np.int8)
    anchor = latent + 0.80 * rng.normal(size=n)
    helpful = latent + 0.35 * rng.normal(size=n) - anchor
    harmful = -helpful
    folds = np.arange(n) % 5
    ids = np.arange(10000, 10000 + n)
    return y, anchor, helpful, harmful, folds, ids


def test_nested_selection_uses_helpful_residual_and_keeps_all_held_folds_positive():
    y, anchor, helpful, _, folds, _ = _fixture()
    result = nested_residual_selection(
        y,
        anchor,
        {"helpful": helpful},
        folds,
        {"helpful": (0.0, 0.10, 0.20, 0.30)},
    )
    assert result["accepted"]
    assert result["honest_gain"] > 0
    assert min(result["held_fold_deltas"]) > 0
    assert result["deploy_weights"]["helpful"] > 0


def test_nested_selection_can_decline_a_harmful_direction():
    y, anchor, _, harmful, folds, _ = _fixture()
    result = nested_residual_selection(
        y,
        anchor,
        {"harmful": harmful},
        folds,
        {"harmful": (0.0, 0.10, 0.20, 0.30)},
    )
    assert result["deploy_weights"]["harmful"] == 0.0
    assert not result["accepted"]


def test_stability_diagnostics_reports_uniform_positive_slices():
    y, anchor, helpful, _, _, ids = _fixture()
    candidate = compose_rank_score(anchor, {"helpful": helpful}, {"helpful": 0.20})
    diagnostics = stability_diagnostics(
        y,
        anchor,
        candidate,
        ids,
        moduli=(2, 3),
        contiguous_blocks=(5,),
    )
    assert all_stability_slices_positive(diagnostics)
    assert diagnostics["modulo"]["2"]["wins"] == 2
    assert diagnostics["contiguous"]["5"]["wins"] == 5


def test_validate_direction_rejects_fold_mismatch():
    oof = np.arange(10, dtype=float)
    test = np.arange(4, dtype=float)
    folds = np.arange(10) % 5
    try:
        validate_direction(
            oof,
            test,
            folds,
            n_train=10,
            n_test=4,
            saved_folds=(folds + 1) % 5,
        )
    except ValueError as exc:
        assert "different fold assignments" in str(exc)
    else:
        raise AssertionError("expected fold mismatch to be rejected")
