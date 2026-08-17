# NOMOPHOBIA 📵

**NOMOPHOBIA: NO MObile PHone PhoBIA**

A falsification-first machine-learning research system for Kaggle Playground Series Season 6 Episode 8, **Predicting Smartphone Addiction**.

> The project name is inspired by *“NOMOPHOBIA: NO MObile PHone PhoBIA”* by Sudip Bhattacharya, Md Abu Bashar, Abhay Srivastava, and Amarjeet Singh (PMCID: PMC6510111; PMID: 31143710).
>
> **Scientific note:** nomophobia and smartphone addiction are related concepts, but they are not interchangeable diagnoses. The Kaggle target is a synthetic binary `addicted_label`; this repository does not attempt to clinically diagnose nomophobia.

## Current thesis

The strongest corrected evidence says this is primarily a **transductive-frequency + behavioral-decomposition** problem, with a deliberately raw model supplying error diversity.

```text
                       user row
                          │
              ┌───────────┴───────────┐
              │                       │
       engineered view             raw view
              │                       │
   behavior + missingness          12 original
   + frequency structure           predictors
   + small digit block                │
              │                       │
       LGBM 63 leaves          LGBM 63 leaves
              │                       │
              └───────────┬───────────┘
                          │
                     rank blend
                          │
                          ▼
                  Kaggle AUC score
```

The raw expert is intentionally **not** decorated with engineered features. Diversity is part of the architecture.

## What the evidence currently says

### Frequency structure is load-bearing

Corrected feature-family ablations indicate approximately:

| Removed family | AUC loss |
|---|---:|
| Full-reference frequency features | ~0.0038 |
| Behavioral decomposition | ~0.0018 |
| Entire digit / rounding family | ~0.0002 |

Early work emphasized decimal artifacts. Mature ablations changed the framing: occurrence density is the dominant engineered signal.

### Capacity mattered more than expected

The early 220-tree screens were underfit. On identical held-out rows, increasing the same `combined63` architecture from 260 to 340 trees produced about **+0.00113 AUC**. Performance continued improving through the 700–1000-tree regime before beginning to flatten.

That invalidated several early diversity kills. CatBoost, XGBoost, and Evidence candidates must be retried at mature capacity before they are rejected.

### The dual-view blend repeatedly survived

Five independent S1 replications were positive:

- 3 replications at 700 trees: mean blend gain ≈ **+0.00079 AUC**
- 2 replications at 1000 trees: mean blend gain ≈ **+0.00105 AUC**

These are independent replications, not the five evaluation subpartitions used inside a single replication. They are promising S1 evidence, not an S3 promotion claim.

## v0.2 production safety rails

The research logic was already careful; the software boundary is now equally strict.

Before training, `load_competition()` validates:

- exact competition columns;
- unique, non-missing IDs;
- binary finite target values;
- finite-or-missing numeric predictors;
- exact `sample_submission.csv` ↔ test ID order.

Every generated submission is then:

1. checked for exact test-row/ID alignment;
2. checked for finite, non-constant ranking scores;
3. written atomically;
4. re-read from disk;
5. validated again;
6. SHA-256 hashed into run metadata.

The goal is simple: a CSV that merely *looks* valid should never escape the pipeline.

See [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md).

## Repository map

```text
nomophobia/
├── README.md
├── src/s6e8/
│   ├── features.py             # behavior, missingness, digit, frequency views
│   ├── frequency.py            # pre-registered frequency sub-family selectors
│   ├── preprocess.py           # shared native categorical preparation
│   ├── models.py               # LightGBM core + lazy optional model families
│   ├── validation.py           # competition/submission contracts
│   ├── submission.py           # rank blending + atomic validated CSV writes
│   ├── artifacts.py            # hashes, atomic manifests, runtime provenance
│   ├── cv.py / evaluate.py     # frozen folds + paired statistical evaluation
│   └── ...                     # ensemble, residual, evidence, meta-model tools
├── experiments/
│   ├── frequency_family_ablation.py
│   ├── frequency_stress.py
│   ├── family_diversity.py
│   ├── evidence_screen.py
│   └── ...
├── scripts/
│   └── make_initial_submission.py
├── tests/
├── docs/
│   ├── REPOSITORY_AUDIT.md
│   ├── SCIENTIFIC_CONTEXT.md
│   ├── ARCHITECTURE.md
│   ├── LEARNINGS.md
│   ├── DECISIONS.md
│   ├── EXECUTION_DIRECTIVE.md
│   ├── KAGGLE_RUNBOOK.md
│   └── VALIDATION.md
├── tune_iterations.py
├── run_winner.py
├── run_s3.py
├── run_diversity_retrial.py
├── stack_external.py
├── select_portfolio.py
└── train.py
```

## Installation

### Production core

For the current LightGBM winner and submission pipeline:

```bash
pip install -e .
```

### Research extras

```bash
# CatBoost + XGBoost diversity retrials
pip install -e ".[diversity]"

# neural experiments
pip install -e ".[neural]"

# all optional model families
pip install -e ".[full]"

# tests
pip install -e ".[dev]"
```

`requirements.txt` remains a convenient full competition environment.

## Expected Kaggle data

Place the competition files in `data/`:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Competition data are not committed to this repository.

## Tests

```bash
pytest -q
```

CI runs the suite on Python 3.10, 3.11, and 3.12 and compiles source/scripts before pytest.

## Generate the initial full-data submission

```bash
python scripts/make_initial_submission.py \
  --data-dir data \
  --out-dir artifacts/submissions/initial_v0_1 \
  --combined-estimators 1000 \
  --raw-estimators 1000 \
  --raw-weight 0.375
```

For byte-level input provenance, add:

```bash
--hash-inputs
```

The primary configuration remains:

```text
0.625 × rank(combined63) + 0.375 × rank(raw63)
```

This is a pragmatic initial submission based on repeated S1 results. It is **not** a substitute for the authoritative S3 campaign.

## Run the production campaign

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

Inspect the generated commands, remove `--dry-run`, and allow the tuned frozen counts to feed the S3 campaign. If tuning lands within 10% of the estimator ceiling, raise the ceiling before S3.

## Highest-EV next feature experiment

Before adding new complexity, decompose the existing frequency gain:

```bash
python experiments/frequency_family_ablation.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --out artifacts/frequency_family_ablation.json \
  --rows 60000 \
  --estimators 700 \
  --folds 5 \
  --device gpu
```

The pre-registered arms are:

- full frequency block;
- no frequency;
- exact-only;
- rounded-only;
- categorical-only;
- exact + rounded.

The non-frequency backbone, model profile, folds, capacity, and full train+test target-free frequency reference remain fixed. This isolates *which density representation is paying rent* before we spend memory on pair/triple frequencies.

## Scientific method

A candidate does not become part of the winner because its AUC is numerically higher once.

Promotion uses:

- identical-row paired comparisons;
- DeLong testing for correlated ROC curves;
- paired bootstrap / Bag of Little Bootstraps at large `n`;
- frozen folds;
- fixed iteration counts during OOF evaluation;
- explicit train/test frequency-reference controls;
- 3 seeds × 5 folds for S3;
- test-missingness-weighted validation;
- held-fold blend-weight selection.

The S3 promotion rule requires at least **13/15 positive fold-seed comparisons**, a positive pooled confidence interval, positive test-like weighted delta, and no unacceptable increase in fold instability.

Negative experiments remain first-class research results. See [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Submission strategy

Treat the daily leaderboard limit as an **experiment budget**:

1. primary dual-view rank blend;
2. mature combined-only control;
3. at most one broad nearby weight probe if the first two separate meaningfully;
4. preserve the remaining slots for S3/public-stack hypotheses.

Public-LB feedback is weak evidence because the public subset introduces sampling noise. OOF remains the source of truth.

## Endgame

After S3, Frontier competes directly against the aligned public OOF universe. Public and Frontier streams are correlation-pruned together before bagged Caruana selection.

The key diagnostic is `frontier_total_weight`:

- **≥15%**: meaningful orthogonal signal
- **5–15%**: marginal contribution; prioritize diversity
- **<5%**: stop tuning this backbone; it lies inside the public solution basin

## Reproducibility status

- audit fixes A1/A4/A5/A7: implemented
- v0.2 competition and submission contracts: implemented
- full 296,302-row inference path: previously verified on v0.1
- full S3 promotion: **pending Kaggle GPU campaign**
- public 74-stream pricing: **pending external OOF input**

## License

MIT. See [`LICENSE`](LICENSE).
