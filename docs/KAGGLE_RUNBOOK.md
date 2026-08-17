# Corrected Kaggle execution runbook

This is the production campaign for NOMOPHOBIA.

## 0. Inputs
Add the official Kaggle competition input containing `train.csv`, `test.csv`, and `sample_submission.csv`. GPU is preferred; Internet is not required for the standalone campaign.

## 1. Stress-test the load-bearing frequency family

```bash
python experiments/frequency_stress.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --rows 120000 \
  --estimators <MATURE_COMBINED_COUNT> \
  --source-rows 60000 \
  --device gpu \
  --out artifacts/frequency_stress_s1.json
```

Escalate if frequency-only train-vs-test adversarial AUC is >0.60; stop and investigate if >0.70. Report train+test minus train-only paired delta, CI, DeLong p, and estimator count.

## 2. Production-scale iteration tuning + S3

Dry-run first:

```bash
python run_winner.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --out-root artifacts/winner \
  --device gpu \
  --tune-rows 628000 \
  --max-estimators 4000 \
  --patience 200 \
  --dry-run
```

Then remove `--dry-run`.

`628000 × 0.88 = 552640` inner-training rows, closely matching one S3 fold (~553095 rows). If either expert's `best_iteration` is >=90% of the estimator ceiling, raise the ceiling and retune before S3.

## 3. S3 reporting contract
For each of seeds 20260816, 20260817, 20260818 report:
- exact fixed estimator count per expert
- expert OOF AUC and fold std
- honest rotating-selection blend AUC
- labeled selection AUC and selection optimism
- rotation weights per held fold
- raw/combined fold-rank correlation
- test-missingness-weighted OOF AUC

`aggregate_promotion.py` enforces the 13/15 promotion rule. `run_s3.py` additionally flags blend-AUC seed spread >0.002 and raw/combined rank correlation >0.988.

## 4. Mature graveyard retrial

Only after S3 freezes the backbone:

```bash
python run_diversity_retrial.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --base-s3-root artifacts/winner/s3 \
  --out-root artifacts/diversity \
  --device gpu \
  --tune-rows 628000 \
  --max-estimators 4000 \
  --patience 200
```

This independently tunes CatBoost and XGBoost, retrains them on the exact S3 fold seeds, regenerates the nonparametric Evidence Expert OOF stream, and forces candidate weights of 5%, 10%, and 15% against each mature S3 blend. A zero optimizer weight is never itself a kill signal.

## 5. Public OOF pricing

```bash
python stack_external.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --base-run artifacts/winner/s3/seed_20260816 \
  --library-dir /kaggle/input/<PUBLIC_OOF_DATASET> \
  --out-dir artifacts/public_price \
  --blend-method caruana
```

ID alignment is mandatory. If the external library exposes folds and they differ from Frontier, rerun Frontier on that fold scheme before any meta model. If it exposes no fold provenance, blend only.

Interpret `frontier_total_weight`:
- >=15%: real orthogonal signal; build portfolio
- 5–15%: marginal; prioritize diversity members
- <5%: stop; Frontier lies inside the public basin

## 6. Final portfolio
Use `select_portfolio.py` after the above. Slot 1 is best honest OOF; Slot 2 is the most robust sufficiently decorrelated submission. Do not spend two slots on >0.999-correlated vectors.
