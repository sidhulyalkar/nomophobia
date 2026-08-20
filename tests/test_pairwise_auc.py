from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.pairwise_auc import (
    fit_pairwise_basis,
    fit_pairwise_ranker,
    score_pairwise_ranker,
    transform_pairwise_basis,
)


def _frame(n: int, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    daily = rng.uniform(2.0, 11.0, n)
    social = rng.uniform(0.0, 6.0, n)
    gaming = rng.uniform(0.0, 4.0, n)
    work = rng.uniform(0.5, 6.0, n)
    sleep = rng.uniform(4.5, 9.0, n)
    notifications = rng.integers(10, 220, n)
    opens = rng.integers(5, 180, n)
    weekend = daily + rng.normal(1.2, 0.8, n)
    age = rng.integers(18, 65, n)
    latent = 0.7 * social + 0.22 * daily + 0.08 * gaming - 0.08 * sleep
    latent += 0.25 * (weekend - daily)
    threshold = np.median(latent)
    y = (latent > threshold).astype(np.int8)
    # Deliberately omit most social-media signal from the champion so the local
    # pairwise ranker has a systematic ordering error to discover.
    champion = daily + 0.15 * gaming + rng.normal(0.0, 0.8, n)
    frame = pd.DataFrame(
        {
            "age": age,
            "daily_screen_time_hours": daily,
            "social_media_hours": social,
            "gaming_hours": gaming,
            "work_study_hours": work,
            "sleep_hours": sleep,
            "notifications_per_day": notifications,
            "app_opens_per_day": opens,
            "weekend_screen_time": weekend,
            "gender": rng.choice(["Male", "Female", "Other"], n),
            "stress_level": rng.choice(["Low", "Medium", "High"], n),
            "academic_work_impact": rng.choice(["No", "Yes"], n),
        }
    )
    return frame, y, champion


def test_pairwise_ranker_learns_local_ordering_signal():
    frame, y, champion = _frame(3000)
    model = fit_pairwise_ranker(
        frame,
        y,
        champion,
        offsets=(1, 2, 4, 8, 16),
        max_pairs=5000,
        alpha=1e-4,
        max_iter=40,
        seed=17,
    )
    score = score_pairwise_ranker(model, frame)
    assert model.pair_total >= 1000
    assert roc_auc_score(y, score) > 0.85


def test_basis_is_finite_with_missing_and_unseen_categories():
    frame, _, _ = _frame(400)
    frame.loc[::9, "social_media_hours"] = np.nan
    frame.loc[::13, "sleep_hours"] = np.nan
    basis = fit_pairwise_basis(frame)
    train_matrix = transform_pairwise_basis(frame, basis)
    probe = frame.iloc[:5].copy()
    probe.loc[:, "gender"] = "Never-Seen"
    probe.loc[:, "daily_screen_time_hours"] = np.nan
    probe_matrix = transform_pairwise_basis(probe, basis)
    assert train_matrix.shape[1] == len(basis.feature_names)
    assert probe_matrix.shape[1] == train_matrix.shape[1]
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(probe_matrix).all()
