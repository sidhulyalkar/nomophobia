from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.generative_evidence import (
    fit_tan_model,
    score_tan_model,
    tan_report,
)


def test_tan_recovers_dependency_signal_that_naive_bayes_cannot():
    # XOR has no univariate class signal.  The label lives entirely in the
    # dependency between the two features, which is exactly what TAN models.
    repeats = 80
    a = np.tile([0.0, 0.0, 1.0, 1.0], repeats)
    b = np.tile([0.0, 1.0, 0.0, 1.0], repeats)
    y = np.tile([0, 1, 1, 0], repeats).astype(np.int8)
    frame = pd.DataFrame({"a": a, "b": b})

    model = fit_tan_model(
        frame,
        y,
        columns=("a", "b"),
        n_bins=2,
        alpha=1.0,
    )
    scores = score_tan_model(model, frame)

    assert roc_auc_score(y, scores["naive"]) == 0.5
    assert roc_auc_score(y, scores["tan"]) > 0.99
    assert len(model.edges) == 1


def test_scores_are_finite_for_missing_and_out_of_range_numeric_values():
    train = pd.DataFrame(
        {
            "a": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 20,
            "b": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0] * 20,
        }
    )
    y = np.tile([0, 0, 0, 1, 1, 1], 20).astype(np.int8)
    model = fit_tan_model(
        train,
        y,
        columns=("a", "b"),
        n_bins=4,
        alpha=1.0,
    )
    probe = pd.DataFrame(
        {"a": [np.nan, -100.0, 100.0], "b": [0.5, np.nan, 2.0]}
    )
    scores = score_tan_model(model, probe)
    assert all(np.isfinite(values).all() for values in scores.values())


def test_report_contains_one_tree_edge_per_non_root_feature():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=500),
            "b": rng.normal(size=500),
            "c": rng.normal(size=500),
            "d": rng.normal(size=500),
        }
    )
    y = ((frame["a"] + 0.5 * frame["b"]) > 0).to_numpy(np.int8)
    model = fit_tan_model(
        frame,
        y,
        columns=("a", "b", "c", "d"),
        n_bins=8,
        alpha=1.0,
    )
    report = tan_report(model)
    assert report["root"] in {"a", "b", "c", "d"}
    assert len(report["edges"]) == 3
    assert len({edge["child"] for edge in report["edges"]}) == 3
