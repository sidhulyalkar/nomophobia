# Validation report

## Audit corrections implemented

- historical XGBoost, Evidence Expert, router, and 31-leaf hedge kills are VOID / RETRIAL REQUIRED because their shared baseline was the 220-tree underfit run;
- CatBoost is explicitly untested historically and has a mature-capacity retrial path;
- zero blend weight is no longer a diversity kill; forced 5/10/15% paired tests are implemented;
- independent replications are distinguished from `eval_subpartitions`;
- scientific framing is frequency-first + behavioral decomposition, not digit-generator-first;
- winner iteration tuning defaults to 628k rows, 4000 maximum estimators, patience 200, with a 90% ceiling stop;
- previously missing graveyard experiment scripts have been restored.

## Post-v3.3 frequency audit

### Source-adversarial safety test

Balanced 100k train + 100k test rows:

- production frequency-only train/test source AUC: **0.56529**
- missingness-only source AUC: **0.56627**
- complete-row frequency-only source AUC: **0.49924**

Interpretation: the measurable source distinguishability is explained by the known missingness shift. There is no evidence here of a hidden complete-row train/test fingerprint in the frequency block.

### Train+test vs train-only frequency reference

60k directional screen, 700-tree `combined63`, disjoint 45k train / 15k evaluation:

- train+test reference AUC: **0.956400**
- train-only reference AUC: **0.955851**
- delta: **+0.000550**
- paired BLB 95% interval: **[+0.000036,+0.001113]**
- 4/5 evaluation subpartitions positive
- DeLong p: ~0.045

This is **S0 directional evidence**, not a promotion. It supports taking the transductive-frequency hypothesis to an S1/mature-capacity Kaggle run.

## Initial full-data submission engineering

The v0.1 initial submission was generated with:

- all **691,369** labeled rows;
- all **296,302** test rows;
- `combined63`: 1000 trees;
- `raw63`: 1000 trees;
- primary blend: 62.5% combined rank / 37.5% raw rank;
- full-data test probability correlation: ~0.98467;
- full-data test rank correlation: ~0.99002;
- 296,302 / 296,302 IDs validated;
- 0 missing/non-finite predictions.

The v0.1 file is a leaderboard probe derived from repeated S1 evidence, **not** an S3-promoted winner.

## Production plumbing

- `run_winner.py --dry-run` resolves production-scale tuning at 628,000 rows;
- `tune_iterations.py` reports training/validation row counts, estimator count, and ceiling status;
- `run_diversity_retrial.py` uses same-seed family training, Evidence OOF, and forced-weight commands;
- `stack_external.py` jointly prunes public + Frontier predictions before reporting Frontier weight;
- full test-schema inference produces 296,302 predictions with 0 missing values.

## Automated checks

**20/20 tests pass** in the development environment.

## Evidence boundary

No S2 or S3 promotion is claimed locally. The next authoritative evidence is production-scale tuning plus the three-seed/five-fold Kaggle GPU campaign.
