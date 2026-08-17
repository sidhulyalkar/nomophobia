# Repository audit and v0.2 hardening

This document records the first repository-level engineering audit after the NOMOPHOBIA v0.1 competition release. The objective is to make the research system harder to misuse without changing the frozen modeling thesis.

## Executive assessment

The repository already has unusually strong scientific hygiene for a Kaggle project: negative results are preserved, promotion gates are explicit, frequency transduction is stress-tested, and the raw-view expert is treated as an intentional diversity mechanism rather than a feature-engineering failure.

The main weakness was different: the **research logic was more mature than the software contract around it**. A valid-looking CSV could still be wrong because IDs were reordered, a competition file had a malformed schema, or predictions contained non-finite values. The full package also forced every optional model family and PyTorch onto every install, even when the production path only needed LightGBM.

v0.2 fixes those issues while leaving the winning feature set, LightGBM profiles, seeds, estimator defaults, and 62.5/37.5 initial blend unchanged.

## High-priority findings

### 1. Submission validity depended too heavily on convention

Previously, `sample_submission.csv` was copied and a prediction vector was assigned. That catches a row-count mismatch, but not reordered IDs or duplicate IDs. A leaderboard artifact can therefore be syntactically valid and semantically wrong.

**Implemented:** `s6e8.validation` now hard-fails on schema errors, duplicate/missing IDs, sample/test order mismatch, non-binary targets, infinite numeric predictors, non-finite submission scores, and constant production predictions.

### 2. Submission generation duplicated preprocessing logic

The production script carried its own categorical-frame helper even though `s6e8.preprocess.prepare_tree_frames` already defined the package contract. Duplication is dangerous because one path can quietly drift from another.

**Implemented:** `scripts/make_initial_submission.py` now uses the shared preprocessing API.

### 3. Reproducibility metadata stopped short of artifact provenance

The old metadata recorded model parameters and correlations but not the exact output hash or runtime package versions.

**Implemented:** every CSV written by the submission helper is re-read, revalidated, SHA-256 hashed, and recorded in `metadata.json`. Runtime package versions and platform metadata are captured. `--hash-inputs` optionally hashes the three Kaggle CSVs for byte-level provenance.

### 4. Optional research dependencies were effectively mandatory

`catboost`, `xgboost`, and `torch` were all base dependencies. `models.py` imported CatBoost and XGBoost at module import time, so even a pure LightGBM submission environment needed them installed.

**Implemented:** CatBoost and XGBoost are lazily imported. The base package now contains only the production dependencies; `diversity`, `neural`, `full`, and `dev` extras are explicit. `requirements.txt` remains the full competition environment.

### 5. CI did not test the package's declared Python range

The package declared Python >=3.10 while CI tested only 3.11.

**Implemented:** CI now tests Python 3.10, 3.11, and 3.12, compiles all source/script files before pytest, caches pip downloads, and uses least-privilege read permissions.

### 6. The next frequency experiment was described but not executable

The corrected research thesis says the highest-EV next feature experiment is to determine which part of the ~0.0038 frequency gain is load-bearing.

**Implemented:** `experiments/frequency_family_ablation.py` pre-registers mutually interpretable frequency arms at mature model capacity:

- full frequency block
- no frequency
- exact-only
- rounded-only
- categorical-only
- exact + rounded

The experiment keeps the non-frequency backbone, folds, model profile, frequency reference population, and estimator count fixed. It is explicitly a directional screen, not promotion evidence.

## Production invariants after v0.2

The following are intentionally unchanged:

- transductive train+test frequency reference for the deployed engineered expert;
- behavioral decomposition features;
- raw 12-feature diversity expert;
- `combined63` and `raw63` LightGBM profiles;
- initial 1000-tree counts;
- initial raw weight 0.375;
- authoritative S3 promotion protocol.

That separation matters. Repository hardening should not smuggle an untested modeling change into the winner.

## Remaining research backlog

The next modeling frontier should proceed in this order:

1. **Run the frequency sub-family ablation at mature capacity.** Determine whether the gain is concentrated in exact counts, rounded density, categorical frequency, or log transforms.
2. **Hierarchical joint-frequency geometry.** Test pair/triple occurrence counts such as `(daily, social)` and `(daily, weekend)` without target means.
3. **Missingness-conditioned density.** Estimate occurrence density within missingness regimes, then repeat the existing source-adversarial safety checks.
4. **Train/test density-ratio features.** Only consider smoothed source-density ratios under a strict safety gate: target gain must survive while complete-row source classification remains near random.
5. **S3 first, library expansion second.** CatBoost/XGBoost/Evidence retrials should use mature capacity and frozen S3 folds rather than reopen the underfit 220-tree graveyard.

## Definition of done for a competition artifact

A production submission is now considered complete only when:

1. the three competition inputs pass the schema/ID contract;
2. both model streams produce finite predictions;
3. blend weights are a valid convex combination;
4. the output CSV is written atomically;
5. the CSV is re-read and validated against test IDs;
6. SHA-256 and runtime metadata are recorded.

This turns “the script finished” into a much stronger statement: **the artifact that left the pipeline is the artifact the pipeline validated.**
