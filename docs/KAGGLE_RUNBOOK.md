# NOMOPHOBIA v0.3 Kaggle execution runbook

The campaign is organized around unresolved questions rather than model-family novelty. GPU is preferred. Internet is not required for the standalone Frontier campaign.

## Input checklist

Official competition input must contain:

```text
train.csv
  691,369 labeled rows expected

test.csv
  296,302 rows expected

sample_submission.csv
  id + addicted_label in exact test order
```

For `original_row_augmentation.py`, add the public source CSV separately. It must contain the 12 raw predictors plus binary `addicted_label`.

## 0. Software verification

```bash
pip install -e ".[dev]"
pytest -q
python -m s6e8 --help
nomophobia validate --data-dir /kaggle/input/playground-series-s6e8
```

## 1. Resolve frequency safety first

```bash
python experiments/frequency_stress.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --rows 120000 \
  --estimators 1000 \
  --source-rows 100000 \
  --device gpu \
  --out artifacts/frequency_stress.json
```

Interpretation:

- target evidence is train+test frequency minus train-only frequency on identical held-out rows;
- overall frequency source AUC is descriptive;
- missingness-only source AUC explains known source shift;
- **complete-row frequency source AUC** is the safety gate.

Warn above 0.55 on complete rows. Stop density expansion above 0.60.

## 2. Decompose marginal frequency

```bash
python experiments/frequency_family_ablation.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --rows 120000 \
  --estimators 1000 \
  --folds 5 \
  --device gpu \
  --out artifacts/frequency_family_ablation.json
```

Use paired intervals. Do not prune a frequency family because its point AUC is microscopically below another arm once.

## 3. Resolve capacity-dependent raw diversity

```bash
python experiments/capacity_diversity_curve.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --rows 180000 \
  --estimators 400 700 1000 1400 2000 \
  --raw-weight 0.375 \
  --device gpu \
  --out artifacts/capacity_diversity_curve.json
```

The blend weight is fixed. This is not a weight search. Watch whether raw/combined rank correlation rises beyond 0.988 and whether fixed-weight paired benefit disappears as capacity increases.

## 4. Probe higher-order frequency geometry

Only proceed if the complete-row source-safety gate is clear.

```bash
python experiments/frequency_geometry.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --rows 120000 \
  --estimators 1000 \
  --device gpu \
  --out-dir artifacts/frequency_geometry
```

Candidate arms include pair/triple density, conditional interaction density, missing-regime-conditioned density, and unsigned train/test support stability. A >+0.003 change is a leak alarm and must not be promoted automatically.

## 5. Optional direct source-row supervision

This is deliberately separate from prior source-prior experiments.

```bash
python experiments/original_row_augmentation.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --original-csv /kaggle/input/<SOURCE>/<source.csv> \
  --rows 120000 \
  --estimators 1000 \
  --weights 0.05 0.10 0.25 0.50 1.0 \
  --device gpu \
  --out-dir artifacts/original_row_augmentation
```

The script removes exact predictor-overlap source rows and evaluates only competition OOF rows. Prefer a broad positive weight region over a single optimized weight.

## 6. Production-scale repeated iteration tuning

You can inspect each expert independently:

```bash
nomophobia tune \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --expert lgb_combined63 \
  --rows 628000 \
  --max-estimators 4000 \
  --patience 200 \
  --repeats 3 \
  --device gpu \
  --out artifacts/tune_combined.json
```

Repeat for `lgb_raw63`, or run the combined winner campaign below. The selected count is the median repeated-holdout best iteration. Stop if any repeat hits the estimator ceiling. Inspect if the iteration range exceeds 30% of its median.

## 7. Authoritative S3 + submission materialization

Dry-run first:

```bash
nomophobia winner \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --out-root artifacts/winner \
  --device gpu \
  --tune-rows 628000 \
  --max-estimators 4000 \
  --patience 200 \
  --tune-repeats 3 \
  --dry-run
```

Then remove `--dry-run`.

S3 runs seeds `20260816`, `20260817`, and `20260818`, each with five frozen folds on all labeled rows.

Mechanical promotion requires:

- 3 distinct 5-fold runs;
- >=13/15 positive fold-seed deltas;
- pooled paired CI above zero;
- positive test-missingness-weighted delta;
- candidate fold standard deviation <=120% of baseline.

Additional v0.3 stop/warning diagnostics:

- seed blend-AUC spread >0.002: stop;
- any production raw/combined rank correlation >0.988: stop and diagnose the raw hedge;
- large rotation-weight std/range: warn about blend instability;
- large selection optimism: warn about weight-selection overfit.

If combined and the dual-view blend both promote, `submission_s3.csv` becomes the seed-bag dual-view candidate. If combined promotes but raw does not, the materialized candidate is combined-only and the research route moves to new diversity rather than tiny raw-weight probes.

## 8. Ask the evidence router what comes next

```bash
nomophobia route \
  --artifact-root artifacts \
  --data-dir /kaggle/input/playground-series-s6e8
```

The router reads completed artifacts and emits the next unresolved high-value action.

## 9. Mature diversity retrial

Only after the S3 backbone is frozen:

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

CatBoost and XGBoost tune independently. Evidence OOF is regenerated. Candidates receive forced 5/10/15% tests. Zero optimized weight is never itself a kill.

## 10. Public OOF pricing

```bash
python stack_external.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --base-run artifacts/winner/s3/seed_20260816 \
  --library-dir /kaggle/input/<PUBLIC_OOF_DATASET> \
  --out-dir artifacts/public_price \
  --blend-method caruana
```

ID alignment is mandatory. If public folds differ, rerun Frontier on those folds before any meta learner. If fold provenance is unavailable, blend only.

Interpret `frontier_total_weight`:

- >=15%: meaningful orthogonal signal;
- 5–15%: marginal; prioritize diversity;
- <5%: stop tuning this backbone.

## 11. Leaderboard budget

Use submissions as hypothesis tests, not an optimizer. Submit the S3 materialized candidate, its meaningful control, and at most one broad alternative when OOF provides a reason. Do not spend slots on adjacent fourth-decimal weights.
