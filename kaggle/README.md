# Kaggle CPU Notebook Runbook

This directory is the operational entry point for running NOMOPHOBIA on Kaggle CPU compute.

The design deliberately uses **two notebooks total**:

1. `01_nomophobia_cpu_research_suite.ipynb` resolves the high-value pre-S3 research questions with cheap gates before expensive branches.
2. `02_nomophobia_cpu_s3_submission.ipynb` freezes iteration counts, runs the authoritative 3-seed × 5-fold S3 campaign, applies the mechanical promotion rule, and emits the submission artifact when a deployable route is promoted.

The notebooks call versioned repository scripts rather than embedding a second copy of the modeling code. This keeps notebook JSON small and prevents the Kaggle path from drifting away from CI-tested source.

## Input matrix

| Notebook | Required inputs | Optional / recommended inputs | Accelerator | Internet |
|---|---|---|---|---|
| 01 CPU Research Suite | Playground Series S6E8 competition input | `guriya79/smart-phone` labeled source dataset | **None (CPU)** | **ON recommended**; OFF supported with `nomophobia-repo-source.zip` |
| 02 CPU S3 + Submission | Playground Series S6E8 competition input | Saved output from Notebook 01 | **None (CPU)** | **ON recommended**; OFF supported with `nomophobia-repo-source.zip` |

Kaggle resource/session limits can change. Check the notebook editor's currently displayed limits before launching the full S3 notebook.

---

# Notebook 01: CPU Research Suite

## Exact input checklist

### Required: competition input

Add the **Playground Series - Season 6, Episode 8** competition as an input.

The notebook expects the standard Kaggle mount:

```text
/kaggle/input/playground-series-s6e8/
├── train.csv
├── test.csv
└── sample_submission.csv
```

The repository validation layer verifies the schema, target, IDs, sample-submission ordering, and finite numeric contract before the experiments start.

### Optional: original labeled source dataset

For the low-weight source-row augmentation experiment, add Kaggle dataset:

```text
guriya79/smart-phone
```

The notebook does **not** depend on a particular CSV filename. It scans non-competition inputs for a CSV containing at least:

```text
addicted_label
daily_screen_time_hours
social_media_hours
gaming_hours
```

The experiment itself requires the full 12-predictor contract and now accepts binary labels expressed as `0/1`, `Yes/No`, `True/False`, or `Addicted/Not Addicted`.

If this dataset is not attached, the source augmentation arm is cleanly skipped. All frequency and diversity experiments still run.

### Repository code input

The easiest mode is:

```text
Internet: ON
```

The notebook clones `sidhulyalkar/nomophobia` and performs an editable `--no-deps` install so it uses Kaggle's preinstalled numerical stack.

For an Internet-OFF notebook, attach the downloadable file:

```text
nomophobia-repo-source.zip
```

The bootstrap cell automatically discovers and extracts it.

## CPU profile selection

The first configuration cell exposes:

```python
PROFILE = "balanced"
```

| Profile | Frequency stress | Family decomposition | Capacity curve | Geometry | Source augmentation | Use case |
|---|---:|---:|---:|---:|---:|---|
| `quick` | 60k rows / 500 trees | 40k / 500 / 3 folds | 60k / 350,600,900 | 40k / 500 / 3 folds | 40k / 500 / 3 folds | plumbing + direction |
| `balanced` | 90k / 700 | 60k / 650 / 3 folds | 100k / 500,800,1200 | 60k / 650 / 3 folds | 60k / 650 / 3 folds | **recommended CPU default** |
| `thorough` | 120k / 1000 | 120k / 1000 / 5 folds | 180k / 400..2000 | 120k / 1000 / 5 folds | 120k / 1000 / 5 folds | mature S1 evidence |

The balanced profile is intentionally not a diluted copy of every experiment. It preserves the high-value structure while using smaller directional folds where the goal is routing, not promotion.

## Synchronous experiment order

The notebook calls `scripts/run_cpu_research_suite.py`, which runs:

```text
frequency safety + transductive reference
              │
              ├── unsafe ──> stop density expansion
              │
              ▼
 marginal frequency decomposition
              │
              ▼
 raw/combined capacity-diversity curve
              │
              ▼
 higher-order density geometry
   (only if frequency safety survives)
              │
              ▼
 optional labeled-source augmentation
```

Important efficiency behavior:

- the three official competition CSVs are SHA-256 hashed **once**;
- child experiments reuse the same data and skip repeated hashing;
- higher-order geometry does not run if the complete-row source-safety gate is red;
- source augmentation does not run when its optional dataset is absent;
- every child experiment remains a standalone JSON + universal manifest for debugging.

## Notebook 01 outputs

The primary outputs are:

```text
/kaggle/working/nomophobia_cpu_research/
├── input_hashes.json
├── frequency_stress.json
├── frequency_family_ablation.json                  # when run
├── capacity_diversity_curve.json
├── frequency_geometry/
│   └── frequency_geometry.json                     # when gated in
├── source_row_augmentation/
│   └── original_row_augmentation.json              # when optional input exists
└── cpu_research_decision.json
```

The notebook also creates:

```text
/kaggle/working/nomophobia_cpu_research_artifacts.zip
```

`cpu_research_decision.json` is the routing artifact to inspect first. It records:

```text
frequency source safety
transductive-frequency decision
high-capacity raw/combined diversity route
advanced geometry arms
advanced source weights
recommended next step
```

## Save Notebook 01

Use **Save Version** with output files enabled.

That saved Notebook output can then be attached directly as an input to Notebook 02. You do not need to manually copy individual JSON files.

---

# Notebook 02: CPU S3 + Submission

## Exact input checklist

### Required: competition input

Same required mount as Notebook 01:

```text
/kaggle/input/playground-series-s6e8/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### Recommended: Notebook 01 output

Add the saved output version of Notebook 01 as a Notebook input.

Notebook 02 recursively searches `/kaggle/input` for:

```text
cpu_research_decision.json
```

If found, it prints the frequency-safety, capacity, geometry, and source-augmentation routing before spending S3 compute.

A `STOP_FREQUENCY_EXPANSION_AUDIT_SOURCE_SHIFT` result blocks the S3 cell deliberately. Other experimental discoveries do **not** alter S3 automatically: S3 still freezes the audited baseline they must eventually beat.

Notebook 02 can run without Notebook 01 when you intentionally want the current audited baseline S3 immediately.

### Repository code

Same bootstrap modes as Notebook 01:

```text
Internet ON  -> clone GitHub repo
Internet OFF -> attach nomophobia-repo-source.zip
```

## Authoritative S3 configuration

Notebook 02 defaults to:

```text
TUNE_ROWS      = 628,000
MAX_ESTIMATORS = 4,000
PATIENCE       = 200
TUNE_REPEATS   = 3
SEEDS          = 20260816, 20260817, 20260818
FOLDS/SEED     = 5
EXPERTS        = combined63 + raw63
DEVICE         = CPU
```

This keeps the scientific S3 definition unchanged. The notebook is operationally optimized, not statistically weakened.

## Restart/resume behavior

The notebook calls `scripts/run_cpu_s3_campaign.py`.

For a reused tuning artifact, the wrapper requires an exact match on:

```text
tune rows
max estimators
patience
repeat count
ceiling fraction
```

It will not silently reuse stale tuning results after a configuration change.

Before spending S3 folds it stops when:

```text
any tuning repeat reaches the estimator ceiling
OR
best-iteration range exceeds 30% of the median
```

Completed S3 seed directories are reused only when their stored expert iteration overrides exactly match the currently frozen combined/raw counts and the required OOF/test/fold artifacts are present.

This makes rerunning a partially completed interactive session much safer.

## Notebook 02 outputs

Authoritative artifacts are written under:

```text
/kaggle/working/nomophobia_cpu_s3/
├── input_hashes.json
├── tune_combined.json
├── tune_raw.json
├── s3/
│   ├── seed_20260816/
│   ├── seed_20260817/
│   ├── seed_20260818/
│   ├── promotion__lgb_raw63__to__lgb_combined63/
│   └── promotion__lgb_combined63__to__blend/
├── cpu_s3_summary.json
├── submission_s3.csv                 # only when a deployable route is promoted
└── submission_s3.json                # validation + SHA-256 record
```

The notebook also creates:

```text
/kaggle/working/nomophobia_cpu_s3_artifacts.zip
```

## S3 deployment outcomes

### Dual-view S3 promotes

The materialized submission is the seed-bagged dual-view candidate:

```text
submission_s3.csv
```

The next research stage is mature CatBoost/XGBoost/Evidence and other orthogonal diversity against this frozen S3 backbone.

### Combined promotes but raw hedge does not

`submission_s3.csv` becomes the seed-bagged combined model. Stop spending submissions on nearby raw-weight grids and prioritize genuinely new diversity.

### S3 does not freeze a deployable candidate

No submission is materialized. Inspect `cpu_s3_summary.json` and the two promotion directories before changing the backbone.

---

# Recommended operating sequence

## First time

1. Import `01_nomophobia_cpu_research_suite.ipynb` into Kaggle.
2. Add the S6E8 competition input.
3. Optionally add `guriya79/smart-phone`.
4. Set Accelerator to **None**.
5. Keep Internet **ON** unless using the offline repo ZIP.
6. Leave `PROFILE="balanced"` for the first real CPU run.
7. Run all cells and save a version with outputs.
8. Inspect `cpu_research_decision.json`.
9. Import `02_nomophobia_cpu_s3_submission.ipynb`.
10. Add the competition input and the saved Notebook 01 output.
11. Run the preflight cells before launching the expensive S3 cell.
12. Run S3 and save output files.
13. If generated, submit the exact `submission_s3.csv`.

## Fast plumbing test

Set Notebook 01:

```python
PROFILE = "quick"
```

Do this before the balanced run if the Kaggle image or input mounting has changed.

## Highest-rigor CPU research pass

After the balanced suite is healthy, change only Notebook 01 to:

```python
PROFILE = "thorough"
```

Do not use `thorough` merely because more compute exists. Its purpose is to confirm promising S1 directions before coding them into the production architecture.

---

# Download bundle layout

The downloadable package accompanying this directory contains:

```text
nomophobia-kaggle-cpu-package/
├── 01_nomophobia_cpu_research_suite.ipynb
├── 02_nomophobia_cpu_s3_submission.ipynb
├── KAGGLE_CPU_GUIDE.md
└── nomophobia-repo-source.zip
```

`nomophobia-repo-source.zip` is only needed for Internet-OFF Kaggle runs. With Internet ON, the notebook clones the latest audited `main` directly.

# What to send back after each run

For Notebook 01, the most useful files are:

```text
cpu_research_decision.json
nomophobia_cpu_research_artifacts.zip
```

For Notebook 02:

```text
cpu_s3_summary.json
nomophobia_cpu_s3_artifacts.zip
submission_s3.csv   # if generated
```

Those artifacts are sufficient to make the next model-design decision without relying on screenshots or copied console logs.
