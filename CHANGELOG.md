# Changelog

## 0.2.0 — Production hardening and frequency decomposition

- Added strict train/test/sample schema and ID-alignment validation.
- Added submission round-trip validation, atomic CSV writes, and SHA-256 artifact hashes.
- Added runtime and optional input-file provenance to submission metadata.
- Refactored the initial submission generator to use shared preprocessing and reusable rank-blend utilities.
- Added CPU/GPU selection and stricter CLI argument validation to submission generation.
- Made CatBoost, XGBoost, and PyTorch optional package extras; LightGBM remains the production core.
- Expanded CI across Python 3.10, 3.11, and 3.12 with source compilation before tests.
- Added regression tests for ID ordering, duplicate IDs, non-finite values, rank blending, CSV round trips, and frequency-family classification.
- Added a mature-capacity frequency sub-family ablation experiment without changing production modeling defaults.
- Added `docs/REPOSITORY_AUDIT.md` documenting the audit, invariants, and next research sequence.

## 0.1.0 — Nomophobia repository launch

- Reframed project around transductive frequency structure + behavioral decomposition.
- Preserved the v3.3 falsification-first research pipeline.
- Added scientific context for the term NOMOPHOBIA.
- Added full-data initial submission generator.
- Added organized documentation for architecture, learnings, decisions, validation, and Kaggle execution.
- Added CI test workflow.
