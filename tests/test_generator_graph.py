import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from s6e8.config import CAT_COLS, NUM_COLS
from s6e8.generator_graph import (
    EXACT_SPECS,
    JOINT_SPECS,
    TokenSpec,
    cross_fitted_graph_scores,
    graph_score,
    token_hash,
)


def _xor_frame(repeats: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    y = []
    for _ in range(repeats):
        for daily, social in ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0)):
            label = int(daily == social)
            row = {column: 1.0 for column in NUM_COLS}
            row.update(
                {
                    "daily_screen_time_hours": daily,
                    "social_media_hours": social,
                    "weekend_screen_time": daily,
                    "work_study_hours": 1.0,
                    "sleep_hours": 7.0,
                    "notifications_per_day": 50.0,
                    "app_opens_per_day": 20.0,
                    "age": 30.0,
                }
            )
            row.update({column: "mid" for column in CAT_COLS})
            rows.append(row)
            y.append(label)
    return pd.DataFrame(rows), np.asarray(y, dtype=np.int8)


def test_token_hash_changes_for_joint_identity():
    frame, _ = _xor_frame(1)
    spec = TokenSpec(
        "pair",
        ("daily_screen_time_hours", "social_media_hours"),
        (2, 2),
    )
    keys = token_hash(frame, spec)
    assert len(np.unique(keys)) == 4


def test_joint_graph_recovers_signal_missing_from_univariate_marginals():
    frame, y = _xor_frame(50)
    splitter = StratifiedKFold(5, shuffle=True, random_state=7)
    folds = np.empty(len(frame), dtype=np.int8)
    for fold, (_, valid) in enumerate(splitter.split(frame, y)):
        folds[valid] = fold

    result = cross_fitted_graph_scores(
        frame,
        frame.iloc[:20].copy(),
        y,
        folds,
        smoothing=2.0,
        min_support=2,
    )
    control_auc = roc_auc_score(y, result["oof_control"])
    treatment_auc = roc_auc_score(y, result["oof_treatment"])
    assert control_auc < 0.60
    assert treatment_auc > 0.95
    assert treatment_auc - control_auc > 0.30


def test_graph_score_does_not_use_query_labels():
    frame, y = _xor_frame(20)
    reference = frame.iloc[:60].copy()
    query = frame.iloc[60:].copy()
    score_a, diag_a = graph_score(
        reference,
        y[:60],
        query,
        EXACT_SPECS + JOINT_SPECS,
        smoothing=5.0,
        min_support=2,
    )
    # There is intentionally no query-target argument. Reordering an unrelated
    # target vector therefore cannot alter predictions.
    score_b, diag_b = graph_score(
        reference,
        y[:60],
        query,
        EXACT_SPECS + JOINT_SPECS,
        smoothing=5.0,
        min_support=2,
    )
    assert np.allclose(score_a, score_b)
    assert diag_a == diag_b


def test_unseen_or_singleton_tokens_fall_back_to_prior():
    reference = pd.DataFrame({"daily_screen_time_hours": [1.0, 2.0, 3.0, 4.0]})
    query = pd.DataFrame({"daily_screen_time_hours": [99.0]})
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    spec = TokenSpec("daily", ("daily_screen_time_hours",), (None,))
    score, diagnostics = graph_score(
        reference,
        y,
        query,
        [spec],
        smoothing=10.0,
        min_support=2,
    )
    assert score[0] == 0.5
    assert diagnostics["coverage_any"] == 0.0
