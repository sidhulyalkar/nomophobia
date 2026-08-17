from __future__ import annotations

from lightgbm import LGBMClassifier


def make_lgb(
    seed: int,
    n_estimators: int,
    profile: str = "raw63",
    monotone_constraints=None,
    device: str = "cpu",
):
    params = dict(
        objective="binary",
        metric="auc",
        n_estimators=n_estimators,
        learning_rate=0.028,
        subsample=0.90,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        max_bin=255,
    )
    profiles = {
        "raw63": dict(num_leaves=63, min_child_samples=100, colsample_bytree=0.90, reg_alpha=0.10),
        "combined63": dict(num_leaves=63, min_child_samples=100, colsample_bytree=0.85, reg_alpha=0.10),
        "semantic15": dict(num_leaves=15, max_depth=5, min_child_samples=120, colsample_bytree=0.90, reg_alpha=0.20),
        "generator31": dict(num_leaves=31, min_child_samples=90, colsample_bytree=0.85, reg_alpha=0.15),
        "monotone31": dict(num_leaves=31, min_child_samples=100, colsample_bytree=0.90, reg_alpha=0.20, monotone_constraints_method="advanced"),
    }
    if profile not in profiles:
        raise ValueError(f"Unknown LightGBM profile: {profile}")
    params.update(profiles[profile])
    if monotone_constraints is not None:
        params["monotone_constraints"] = monotone_constraints
    if device == "gpu":
        params["device_type"] = "gpu"
    elif device != "cpu":
        raise ValueError(f"Unknown device: {device}")
    return LGBMClassifier(**params)


def make_cat(seed: int, iterations: int, profile: str = "raw", device: str = "cpu"):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is an optional dependency. Install with "
            '`pip install -e ".[diversity]"` or `pip install -r requirements.txt`.'
        ) from exc

    return CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", iterations=iterations,
        learning_rate=0.035, depth=8, l2_leaf_reg=6.0,
        random_seed=seed, random_strength=0.35, border_count=254,
        bootstrap_type="Bayesian", bagging_temperature=0.5,
        allow_writing_files=False, verbose=False, thread_count=-1,
        task_type="GPU" if device == "gpu" else "CPU",
    )


def make_xgb(seed: int, n_estimators: int, profile: str = "raw", device: str = "cpu"):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is an optional dependency. Install with "
            '`pip install -e ".[diversity]"` or `pip install -r requirements.txt`.'
        ) from exc

    return XGBClassifier(
        objective="binary:logistic", eval_metric="auc", n_estimators=n_estimators,
        learning_rate=0.028, max_depth=7, min_child_weight=10,
        subsample=0.88, colsample_bytree=0.90,
        reg_alpha=0.08, reg_lambda=2.2, random_state=seed,
        n_jobs=-1, tree_method="hist", max_bin=256,
        enable_categorical=True,
        device="cuda" if device == "gpu" else "cpu",
    )


def focal_binary_objective(alpha: float = 0.25, gamma: float = 2.0):
    import numpy as np
    a = float(alpha); g = float(gamma)
    if not (0.0 < a < 1.0) or g < 0:
        raise ValueError("alpha must be in (0,1) and gamma >= 0")

    def objective(y_true, raw_pred):
        z = np.clip(np.asarray(raw_pred, float), -18, 18)
        p = 1.0 / (1.0 + np.exp(-z)); q = 1.0 - p
        y = np.asarray(y_true, float)
        pc = np.clip(p, 1e-8, 1-1e-8); qc = 1.0-pc
        b1 = g * pc * np.log(pc) - qc
        grad1 = a * (qc ** g) * b1
        hess1 = a * pc * (qc ** g) * (-g * b1 + qc * (g * (np.log(pc) + 1.0) + 1.0))
        b0 = pc - g * qc * np.log(qc)
        grad0 = (1.0-a) * (pc ** g) * b0
        hess0 = (1.0-a) * qc * (pc ** g) * (g * b0 + pc * (1.0 + g * (np.log(qc) + 1.0)))
        grad = np.where(y > 0.5, grad1, grad0)
        hess = np.maximum(np.where(y > 0.5, hess1, hess0), 1e-7)
        return grad, hess
    return objective


def make_lgb_focal(seed: int, n_estimators: int, profile: str = "raw63",
                   alpha: float = 0.25, gamma: float = 2.0, device: str = "cpu"):
    model = make_lgb(seed, n_estimators, profile, device=device)
    model.set_params(objective=focal_binary_objective(alpha, gamma), metric="auc")
    return model
