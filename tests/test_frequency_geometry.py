import numpy as np
import pandas as pd

from s6e8.frequency_geometry import add_joint_density_features, add_regime_conditioned_density_features, add_source_stability_features


def _raw(n=8):
    return pd.DataFrame({
        "age": [20, 21, 20, 22, 20, 23, 24, 20][:n],
        "daily_screen_time_hours": [8.1, 6.2, 8.1, 4.5, 8.1, 9.0, 5.5, 8.1][:n],
        "social_media_hours": [3.2, 2.0, 3.2, 1.0, 3.2, 4.0, 1.5, 3.2][:n],
        "gaming_hours": [2.0, 1.0, 2.0, 0.5, 2.0, 2.5, 1.0, 2.0][:n],
        "work_study_hours": [2.5, 3.0, 2.5, 5.0, 2.5, 2.0, 4.0, 2.5][:n],
        "sleep_hours": [7.0, 8.0, 7.0, 8.2, 7.0, 6.0, 7.5, 7.0][:n],
        "notifications_per_day": [100, 80, 100, 40, 100, 160, 60, 100][:n],
        "app_opens_per_day": [50, 40, 50, 20, 50, 90, 30, 50][:n],
        "weekend_screen_time": [10.4, 7.0, 10.4, 5.0, 10.4, 11.2, 6.0, 10.4][:n],
        "gender": ["Female", "Male", "Female", "Male", "Female", "Other", "Male", "Female"][:n],
        "stress_level": ["High", "Low", "High", "Medium", "High", "High", "Low", "High"][:n],
        "academic_work_impact": ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes"][:n],
    })


def test_joint_density_repeated_state_gets_repeated_support():
    tr = _raw(6); te = _raw(2)
    a, b, meta = add_joint_density_features(tr, te, reference_train=tr, reference_test=te)
    assert meta["reference_rows"] == len(tr) + len(te)
    col = "density__daily_social__freq"
    assert float(a.loc[0, col]) > 1
    assert np.isfinite(a.to_numpy(float)).all()
    assert "density__daily_social__interaction" in a


def test_regime_density_and_source_stability_are_finite():
    tr = _raw(8); te = _raw(4)
    tr.loc[1, "gaming_hours"] = np.nan
    te.loc[2, "social_media_hours"] = np.nan
    rtr, rte, _ = add_regime_conditioned_density_features(tr, te, reference_train=tr, reference_test=te)
    str_, ste, _ = add_source_stability_features(tr, te, reference_train=tr, reference_test=te)
    assert rtr.shape[0] == len(tr) and rte.shape[0] == len(te)
    assert np.isfinite(rtr.to_numpy(float)).all()
    assert np.isfinite(str_.to_numpy(float)).all()
    balance = str_.filter(like="_balance").to_numpy(float)
    assert (balance >= 0).all() and (balance <= 1).all()
    assert ste.shape[1] == str_.shape[1]
