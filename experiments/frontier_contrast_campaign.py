#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8.config import ID_COL, TARGET
from s6e8.contrast import ResidualGate, apply_rank_residual, rank_direction, rotating_residual_gate
from s6e8.identity import build_contrast_feature_frame, categorical_feature_names
from s6e8.submission import write_submission
from s6e8.validation import validate_competition_frames

ROUND_DEFAULTS = {"lgb": 900, "xgb": 1500, "cat": 4000}
DEFAULT_WEIGHT_GRID = [
    0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015,
    0.020, 0.025, 0.030, 0.040, 0.050, 0.065, 0.080,
]


def _parse_weights(text: str | None) -> list[float]:
    if text is None:
        return DEFAULT_WEIGHT_GRID.copy()
    weights = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not weights:
        raise ValueError("weight grid is empty")
    if weights[0] != 0.0:
        weights = [0.0, *weights]
    if any(b <= a for a, b in zip(weights, weights[1:])):
        raise ValueError("weights must be strictly increasing")
    return weights


def _read_competition(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    validate_competition_frames(train, test, sample)
    return train, test, sample


def _load_anchor(
    anchor_oof_path: Path,
    anchor_test_path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    oof_col: str,
    test_col: str,
    fold_col: str,
):
    oof = pd.read_csv(anchor_oof_path)
    pred_test = pd.read_csv(anchor_test_path)
    for required in (ID_COL, fold_col, oof_col):
        if required not in oof:
            raise ValueError(f"anchor OOF is missing required column {required!r}")
    for required in (ID_COL, test_col):
        if required not in pred_test:
            raise ValueError(f"anchor test is missing required column {required!r}")
    if len(oof) != len(train) or not np.array_equal(
        oof[ID_COL].to_numpy(), train[ID_COL].to_numpy()
    ):
        raise ValueError("anchor OOF IDs are not aligned to train.csv")
    if len(pred_test) != len(test) or not np.array_equal(
        pred_test[ID_COL].to_numpy(), test[ID_COL].to_numpy()
    ):
        raise ValueError("anchor test IDs are not aligned to test.csv")
    if TARGET in oof:
        y_file = pd.to_numeric(oof[TARGET]).to_numpy(np.int8)
        if not np.array_equal(y_file, train[TARGET].to_numpy(np.int8)):
            raise ValueError("anchor OOF target column does not match train.csv")
    anchor_oof = pd.to_numeric(oof[oof_col]).to_numpy(float)
    anchor_test = pd.to_numeric(pred_test[test_col]).to_numpy(float)
    folds = pd.to_numeric(oof[fold_col]).to_numpy(int)
    if not np.isfinite(anchor_oof).all() or not np.isfinite(anchor_test).all():
        raise ValueError("anchor predictions contain non-finite values")
    return anchor_oof, anchor_test, folds


def _combined_categories(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    categories = {}
    for column in columns:
        values = pd.concat(
            [train[column].astype("string"), test[column].astype("string")],
            ignore_index=True,
        ).fillna("__MISSING__")
        categories[column] = sorted(values.unique().tolist())
    return categories


def _prepare_lgb(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    cat_cols = sorted(
        set(categorical_feature_names(train)) | set(categorical_feature_names(test))
    )
    categories = _combined_categories(train, test, cat_cols)
    frames = []
    for frame in (train.copy(), valid.copy(), test.copy()):
        for column in cat_cols:
            frame[column] = pd.Categorical(
                frame[column].astype("string").fillna("__MISSING__"),
                categories=categories[column],
            )
        frames.append(frame)
    return (*frames, cat_cols)


def _prepare_codes(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    cat_cols = sorted(
        set(categorical_feature_names(train)) | set(categorical_feature_names(test))
    )
    categories = _combined_categories(train, test, cat_cols)
    frames = []
    for frame in (train.copy(), valid.copy(), test.copy()):
        for column in cat_cols:
            category = pd.Categorical(
                frame[column].astype("string").fillna("__MISSING__"),
                categories=categories[column],
            )
            frame[column] = (
                pd.Series(category.codes, index=frame.index)
                .replace(-1, np.nan)
                .astype(np.float32)
            )
        frames.append(frame.astype(np.float32))
    return (*frames, [])


def _prepare_cat(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    cat_cols = sorted(
        set(categorical_feature_names(train)) | set(categorical_feature_names(test))
    )
    frames = []
    for frame in (train.copy(), valid.copy(), test.copy()):
        for column in cat_cols:
            frame[column] = (
                frame[column].astype("string").fillna("__MISSING__").astype(str)
            )
        frames.append(frame)
    return (*frames, cat_cols)


def _fit_predict(
    family: str,
    train: pd.DataFrame,
    y: np.ndarray,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
    rounds: int,
    device: str,
):
    """Fit a fixed schedule. Outer validation is never used for checkpoint selection."""
    if family == "lgb":
        import lightgbm as lgb

        tr, va, te, cat_cols = _prepare_lgb(train, valid, test)
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=rounds,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=100,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.90,
            reg_alpha=0.10,
            reg_lambda=2.0,
            max_bin=255,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            device_type="gpu" if device == "gpu" else "cpu",
        )
        model.fit(tr, y, categorical_feature=cat_cols)
    elif family == "xgb":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError('Install the diversity extra: pip install -e ".[diversity]"') from exc

        tr, va, te, _ = _prepare_codes(train, valid, test)
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=rounds,
            learning_rate=0.035,
            max_depth=8,
            min_child_weight=20.0,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.10,
            reg_lambda=2.0,
            max_bin=256,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            device="cuda" if device == "gpu" else "cpu",
        )
        model.fit(tr, y, verbose=False)
    elif family == "cat":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError('Install the diversity extra: pip install -e ".[diversity]"') from exc

        tr, va, te, cat_cols = _prepare_cat(train, valid, test)
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=rounds,
            learning_rate=0.035,
            depth=8,
            l2_leaf_reg=6.0,
            random_seed=seed,
            random_strength=0.35,
            border_count=254,
            bootstrap_type="Bayesian",
            bagging_temperature=0.5,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
            task_type="GPU" if device == "gpu" else "CPU",
        )
        model.fit(tr, y, cat_features=cat_cols)
    else:
        raise ValueError(f"unknown family: {family}")

    return model.predict_proba(va)[:, 1], model.predict_proba(te)[:, 1]


def _load_control_artifacts(
    path: Path,
    *,
    n_train: int,
    n_test: int,
    family: str,
    rounds: int,
    folds: np.ndarray,
):
    meta_path = path / "control_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} does not exist")
    meta = json.loads(meta_path.read_text())
    expected = {"family": family, "rounds": int(rounds), "rows": int(n_train)}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"control artifact mismatch for {key}: expected {value!r}, got {meta.get(key)!r}"
            )
    saved_folds = np.load(path / "folds.npy")
    if not np.array_equal(saved_folds, folds):
        raise ValueError("control artifacts use different fold assignments")
    oof = np.load(path / "oof_control.npy")
    test = np.load(path / "test_control.npy")
    if len(oof) != n_train or len(test) != n_test:
        raise ValueError("control prediction lengths do not match competition data")
    return np.asarray(oof, float), np.asarray(test, float)


def main():
    parser = argparse.ArgumentParser(
        description="Matched fixed-schedule residual campaign against an aligned frontier anchor."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--anchor-oof", required=True)
    parser.add_argument("--anchor-test", required=True)
    parser.add_argument("--anchor-oof-col", default="honest_blend")
    parser.add_argument("--anchor-test-col", default=TARGET)
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--family", choices=sorted(ROUND_DEFAULTS), default="lgb")
    parser.add_argument(
        "--treatment",
        choices=["identity", "screen", "identity_screen"],
        default="identity_screen",
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--min-fold-wins", type=int, default=4)
    parser.add_argument("--fold-tolerance", type=float, default=-2e-6)
    parser.add_argument("--slice-tolerance", type=float, default=-2e-6)
    parser.add_argument("--reuse-control-dir", default=None)
    parser.add_argument("--out-dir", default="artifacts/frontier_contrast")
    args = parser.parse_args()

    started = time.time()
    rounds = int(args.rounds or ROUND_DEFAULTS[args.family])
    weights = _parse_weights(args.weights)
    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, test, sample = _read_competition(data_dir)
    y = train[TARGET].to_numpy(np.int8)
    ids = train[ID_COL].to_numpy()
    anchor_oof, anchor_test, folds = _load_anchor(
        Path(args.anchor_oof),
        Path(args.anchor_test),
        train,
        test,
        oof_col=args.anchor_oof_col,
        test_col=args.anchor_test_col,
        fold_col=args.fold_col,
    )
    unique_folds = np.unique(folds)
    if len(unique_folds) < 3:
        raise ValueError("at least three aligned outer folds are required")

    include_exact_categories = args.family == "cat"
    control_full = build_contrast_feature_frame(train, "raw")
    control_test = build_contrast_feature_frame(test, "raw")
    treatment_full = build_contrast_feature_frame(
        train, args.treatment, include_exact_categories=include_exact_categories
    )
    treatment_test = build_contrast_feature_frame(
        test, args.treatment, include_exact_categories=include_exact_categories
    )

    if args.reuse_control_dir:
        oof_control, test_control = _load_control_artifacts(
            Path(args.reuse_control_dir),
            n_train=len(train),
            n_test=len(test),
            family=args.family,
            rounds=rounds,
            folds=folds,
        )
        train_control = False
    else:
        oof_control = np.empty(len(train), dtype=float)
        test_control = np.zeros(len(test), dtype=float)
        train_control = True

    oof_treatment = np.empty(len(train), dtype=float)
    test_treatment = np.zeros(len(test), dtype=float)
    fold_metrics = []

    for fold in unique_folds:
        t0 = time.time()
        train_idx = np.where(folds != fold)[0]
        valid_idx = np.where(folds == fold)[0]
        yy = y[train_idx]
        yv = y[valid_idx]

        if train_control:
            pred_control, test_part_control = _fit_predict(
                args.family,
                control_full.iloc[train_idx].reset_index(drop=True),
                yy,
                control_full.iloc[valid_idx].reset_index(drop=True),
                control_test.reset_index(drop=True),
                seed=args.seed + int(fold),
                rounds=rounds,
                device=args.device,
            )
            oof_control[valid_idx] = pred_control
            test_control += test_part_control / len(unique_folds)
        else:
            pred_control = oof_control[valid_idx]

        pred_treatment, test_part_treatment = _fit_predict(
            args.family,
            treatment_full.iloc[train_idx].reset_index(drop=True),
            yy,
            treatment_full.iloc[valid_idx].reset_index(drop=True),
            treatment_test.reset_index(drop=True),
            seed=args.seed + int(fold),
            rounds=rounds,
            device=args.device,
        )
        oof_treatment[valid_idx] = pred_treatment
        test_treatment += test_part_treatment / len(unique_folds)

        control_auc = float(roc_auc_score(yv, pred_control))
        treatment_auc = float(roc_auc_score(yv, pred_treatment))
        row = {
            "fold": int(fold),
            "control_auc": control_auc,
            "treatment_auc": treatment_auc,
            "treatment_minus_control": treatment_auc - control_auc,
            "rank_corr": float(
                pd.Series(pred_control).rank(method="average").corr(
                    pd.Series(pred_treatment).rank(method="average")
                )
            ),
            "seconds": round(time.time() - t0, 2),
        }
        fold_metrics.append(row)
        print(json.dumps(row), flush=True)

    direction_oof = rank_direction(oof_treatment, oof_control)
    direction_test = rank_direction(test_treatment, test_control)
    gate = ResidualGate(
        min_gain=args.min_gain,
        min_fold_wins=args.min_fold_wins,
        fold_tolerance=args.fold_tolerance,
        require_id_slices=True,
        slice_tolerance=args.slice_tolerance,
    )
    decision = rotating_residual_gate(
        y, anchor_oof, direction_oof, folds, weights, ids=ids, gate=gate
    )
    honest = decision.pop("honest_oof")
    raw_weight = float(decision["selected_weight_median"])
    raw_candidate_oof = apply_rank_residual(anchor_oof, direction_oof, raw_weight)
    raw_candidate_test = apply_rank_residual(anchor_test, direction_test, raw_weight)
    gated_test = (
        apply_rank_residual(
            anchor_test, direction_test, float(decision["deploy_weight"])
        )
        if decision["accepted"]
        else anchor_test.copy()
    )

    np.save(out / "folds.npy", folds)
    np.save(out / "oof_control.npy", oof_control)
    np.save(out / "oof_treatment.npy", oof_treatment)
    np.save(out / "test_control.npy", test_control)
    np.save(out / "test_treatment.npy", test_treatment)
    np.save(out / "direction_oof.npy", direction_oof)
    np.save(out / "direction_test.npy", direction_test)
    np.save(out / "oof_honest_residual.npy", honest)
    np.save(out / "oof_raw_deploy_candidate.npy", raw_candidate_oof)

    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "fold": folds,
            "anchor": anchor_oof,
            "control": oof_control,
            "treatment": oof_treatment,
            "direction": direction_oof,
            "honest_residual": honest,
            "raw_deploy_candidate": raw_candidate_oof,
        }
    ).to_csv(out / "oof.csv", index=False)

    candidate_stats = write_submission(
        out / "submission_candidate.csv", sample, test, raw_candidate_test
    )
    gated_stats = write_submission(
        out / "submission_gated.csv", sample, test, gated_test
    )

    control_meta = {
        "family": args.family,
        "rounds": rounds,
        "rows": len(train),
        "seed": args.seed,
        "folds": [int(value) for value in unique_folds],
        "feature_set": "raw",
    }
    (out / "control_meta.json").write_text(json.dumps(control_meta, indent=2) + "\n")

    report = {
        "version": "frontier-contrast-v1",
        "family": args.family,
        "treatment": args.treatment,
        "rounds": rounds,
        "seed": args.seed,
        "device": args.device,
        "weights": weights,
        "features": {
            "control": int(control_full.shape[1]),
            "treatment": int(treatment_full.shape[1]),
            "include_exact_categories": include_exact_categories,
        },
        "standalone": {
            "anchor_auc": float(roc_auc_score(y, anchor_oof)),
            "control_auc": float(roc_auc_score(y, oof_control)),
            "treatment_auc": float(roc_auc_score(y, oof_treatment)),
            "treatment_minus_control": float(
                roc_auc_score(y, oof_treatment) - roc_auc_score(y, oof_control)
            ),
            "control_treatment_rank_corr": float(
                pd.Series(oof_control).rank(method="average").corr(
                    pd.Series(oof_treatment).rank(method="average")
                )
            ),
        },
        "fold_metrics": fold_metrics,
        "residual_gate": decision,
        "candidate_submission": candidate_stats,
        "gated_submission": gated_stats,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "decision.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
