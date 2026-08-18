# Changelog

## 0.3.2 — Live target-encoding frontier

- Added a full-data leakage-safe exact-value target-encoding frontier with LightGBM and XGBoost, including rotating held-fold rank-blend selection.
- Added replicated 120k screens for target-encoding uncertainty metadata, inner-fold count, small-sample capacity, and smoothing diversity.
- Promoted smoothing-20 only as a diversity stream after its 50/50 blend with smoothing-10 improved all three replicated screens.
- Added full-data smoothing-20 LightGBM and an aligned four-stream live blend across inner-5 LightGBM, XGBoost, inner-10 LightGBM, and smoothing-20 LightGBM.
- Added deterministic submission materialization with row/ID/finite-score validation and SHA-256 metadata.
- Added a manual GitHub Actions workflow for explicitly running the live frontier lanes without auto-triggering expensive research jobs.
- Added `docs/LIVE_FRONTIER_2026-08-17.md` with the exact evidence, fold metrics, blend weights, commands, and submission hashes.

## 0.3.1 — Kaggle CPU notebook operations

- Added a two-notebook Kaggle CPU workflow under `kaggle/notebooks/` to minimize manual orchestration.
- Added a gated CPU research suite that runs frequency safety, marginal decomposition, capacity diversity, higher-order density geometry, and optional source augmentation synchronously.
- Added a resumable CPU S3 wrapper that refuses stale tuning artifacts, stops on estimator-ceiling or tuning-instability warnings, reuses only matching completed seed runs, and materializes the promoted submission.
- Added one-time competition input hashing to avoid repeated SHA-256 work across child experiments.
- Added a detailed Kaggle input/output runbook with exact mounts, CPU profiles, online/offline repository bootstrap, notebook chaining, and artifact handoff.
- Added CI checks that parse every Kaggle notebook code cell with Python's AST.
- Made source-row augmentation accept common binary label encodings such as `0/1`, `Yes/No`, and `True/False` while rejecting ambiguous labels.

## 0.3.0 — Frequency geometry and authoritative S3 engine

- Added installable `nomophobia` CLI with `validate`, `route`, `tune`, `s3`, and `winner` commands.
- Added universal failure-safe experiment manifests containing scientific contract, Git/runtime provenance, input hashes, elapsed time, metrics, and output hashes.
- Replaced one-shot iteration calibration with repeated production-scale holdouts and a median frozen estimator count, including ceiling and instability guards.
- Added S3 diagnostics for seed spread, production raw/combined rank correlation, blend-selection optimism, and held-fold weight stability.
- Added automatic S3 routing and validated materialization of the promoted seed-bag submission when the promotion gates pass.
- Deepened frequency safety controls into overall frequency source AUC, missingness-only source AUC, and complete-row frequency source AUC.
- Upgraded frequency-family ablations to paired OOF comparisons.
- Added selected pair/triple frequency geometry, missing-regime-conditioned density, and unsigned train/test support-stability features.
- Added a fixed-weight capacity/diversity curve to test whether the raw hedge collapses as model capacity rises.
- Added direct low-weight augmentation from the labeled source dataset after exact predictor-overlap removal.
- Added an evidence router that converts completed artifacts into the next highest-value experiment rather than accumulating ad hoc notebooks.
- Preserved the current production model until new candidates survive the existing evidence gates.

## 0.2.0 — Production hardening and frequency decomposition

- Added strict train/test/sample schema and ID-alignment validation.
- Added submission round-trip validation, atomic CSV writes, and SHA-256 artifact hashes.
- Added runtime and optional input-file provenance to submission metadata.
- Refactored the initial submission generator to use shared preprocessing and reusable rank-blend utilities.
- Made CatBoost, XGBoost, and PyTorch optional package extras; LightGBM remains the production core.
- Expanded CI across Python 3.10, 3.11, and 3.12.
- Added a mature-capacity frequency sub-family ablation and repository audit.

## 0.1.0 — Nomophobia repository launch

- Reframed project around transductive frequency structure + behavioral decomposition.
- Preserved the falsification-first research pipeline and append-only decision ledger.
- Added scientific context, full-data submission generation, organized documentation, and CI.
