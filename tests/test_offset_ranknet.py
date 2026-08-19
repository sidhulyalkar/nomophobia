from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.offset_ranknet import fit_offset_ranknet, score_offset_ranknet


def test_offset_ranknet_improves_a_champion_missing_one_signal():
    rng = np.random.default_rng(23)
    n = 1800
    daily = rng.uniform(2, 11, n)
    social = rng.uniform(0, 6, n)
    gaming = rng.uniform(0, 4, n)
    latent = 0.55 * daily + 1.15 * social + 0.20 * gaming + rng.normal(0, 0.7, n)
    y = (latent > np.median(latent)).astype(np.int8)
    champion = 0.75 * daily + 0.10 * gaming + rng.normal(0, 0.5, n)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 65, n),
            "daily_screen_time_hours": daily,
            "social_media_hours": social,
            "gaming_hours": gaming,
            "work_study_hours": rng.uniform(1, 7, n),
            "sleep_hours": rng.uniform(4.5, 9, n),
            "notifications_per_day": rng.integers(10, 220, n),
            "app_opens_per_day": rng.integers(5, 180, n),
            "weekend_screen_time": daily + rng.normal(1, 0.8, n),
            "gender": rng.choice(["Male", "Female", "Other"], n),
            "stress_level": rng.choice(["Low", "Medium", "High"], n),
            "academic_work_impact": rng.choice(["No", "Yes"], n),
        }
    )
    model = fit_offset_ranknet(
        frame,
        y,
        champion,
        max_pairs=12000,
        offsets=(8, 32, 128, 512),
        anchor_temperature=6.0,
        l2=0.003,
        max_iter=18,
        seed=5,
    )
    score = score_offset_ranknet(model, frame, champion)
    assert np.isfinite(score).all()
    assert roc_auc_score(y, score) > roc_auc_score(y, champion) + 0.02
