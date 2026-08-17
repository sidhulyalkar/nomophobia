# Changelog

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
