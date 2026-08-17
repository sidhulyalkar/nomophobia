# NOMOPHOBIA 📵

**NOMOPHOBIA: NO MObile PHone PhoBIA**

A falsification-first machine-learning research project for Kaggle Playground Series Season 6 Episode 8, **Predicting Smartphone Addiction**.

> The project name is inspired by *“NOMOPHOBIA: NO MObile PHone PhoBIA”* by Sudip Bhattacharya, Md Abu Bashar, Abhay Srivastava, and Amarjeet Singh (PMCID: PMC6510111; PMID: 31143710).
>
> **Scientific note:** nomophobia and smartphone addiction are related concepts, but they are not interchangeable diagnoses. The Kaggle competition target is a synthetic binary `addicted_label`; this repository does not attempt to clinically diagnose nomophobia.

## Why this repository exists

The goal is not to accumulate notebooks until one lands on a lucky public-leaderboard decimal. The goal is to build a reproducible competition system that can answer:

1. What information is actually present in the data?
2. Which features survive paired out-of-fold testing?
3. Which models make genuinely different errors?
4. Which apparent wins disappear when sample size or model capacity changes?
5. Does our final prediction stream add information beyond the public ensemble ecosystem?

The project therefore treats **negative experiments as first-class results** and uses explicit promotion gates.

## Current thesis

The strongest corrected evidence says this is primarily a:

**transductive-frequency + behavioral-decomposition problem**, with a deliberately raw model supplying useful error diversity.

The current winner candidate is a two-view high-capacity LightGBM system:

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
                   fold-wise ranks
                          │
              honest held-fold blending
                          │
                          ▼
                    final AUC score
```

The raw expert is intentionally *not* decorated with the engineered features. Diversity is part of the architecture.

## What we learned

### 1. Frequency structure is load-bearing

Corrected feature-family ablations indicate approximately:

| Removed family | AUC loss |
|---|---:|
| Full-reference frequency features | ~0.0038 |
| Behavioral decomposition | ~0.0018 |
| Entire digit / rounding family | ~0.0002 |

This changed the research framing. Early work emphasized decimal artifacts; later ablations showed that they are only a small contributor.

### 2. Capacity mattered much more than expected

The early 220-tree screens were underfit. On identical held-out rows, increasing the same `combined63` architecture from 260 to 340 trees produced about **+0.00113 AUC**. Performance continued improving through 700–1000 trees before beginning to flatten.

### 3. The dual-view blend repeatedly survived

Five independent S1 replications were positive:

- 3 replications at 700 trees: mean blend gain ≈ **+0.00079 AUC**
- 2 replications at 1000 trees: mean blend gain ≈ **+0.00105 AUC**

These are **independent replications**, not the five evaluation subpartitions used inside each replication. They are promising S1 evidence, not an S3 promotion claim.

### 4. Attractive ideas have been killed

Examples include naive missing-regime specialists, regime isotonic calibration, multiple-imputation marginalization, joint digit tuples, explicit coarse target-posterior tables, rank objectives, and focal-loss variants. Several earlier diversity kills were later voided because they had been run at only 220 trees; mature-capacity CatBoost/XGBoost/Evidence retrials are now part of the formal campaign.

See [`docs/DECISIONS.md`](docs/DECISIONS.md) for the append-only research ledger.

## Scientific method

A candidate does not become part of the winner merely because its AUC is numerically higher once.

Promotion uses:

- identical-row paired comparisons
- DeLong testing for correlated ROC curves
- paired bootstrap / Bag of Little Bootstraps at large `n`
- frozen folds
- fixed iteration counts during OOF evaluation
- explicit train/test frequency-reference controls
- 3 seeds × 5 folds for S3
- test-missingness-weighted validation
- held-fold blend-weight selection

The S3 promotion rule requires at least **13/15 positive fold-seed comparisons**, a positive pooled confidence interval, positive test-like weighted delta, and no unacceptable increase in fold instability.

## Repository map

```text
nomophobia/
├── README.md
├── src/s6e8/                  # reusable feature/model/evaluation package
├── experiments/               # falsifiable research experiments
├── scripts/
│   └── make_initial_submission.py
├── tests/                     # regression + audit tests
├── docs/
│   ├── SCIENTIFIC_CONTEXT.md
│   ├── ARCHITECTURE.md
│   ├── LEARNINGS.md
│   ├── DECISIONS.md
│   ├── EXECUTION_DIRECTIVE.md
│   ├── KAGGLE_RUNBOOK.md
│   └── VALIDATION.md
├── run_winner.py              # production-scale tuning + S3 orchestration
├── run_s3.py                  # 3-seed × 5-fold campaign
├── run_diversity_retrial.py   # mature Cat/XGB/Evidence forced-weight trial
├── stack_external.py          # public OOF pricing
├── select_portfolio.py        # final submission portfolio
└── train.py                   # general CV runner
```

## Quick start

### Install

```bash
pip install -r requirements.txt
```

### Expected data

Place Kaggle's files in `data/`:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Competition data are **not** committed to this repository.

### Run tests

```bash
pytest -q
```

### Generate an initial full-data submission

```bash
python scripts/make_initial_submission.py \
  --data-dir data \
  --out-dir artifacts/submissions/initial_v0_1 \
  --combined-estimators 1000 \
  --raw-estimators 1000 \
  --raw-weight 0.375
```

This is a pragmatic initial submission based on repeated S1 results. It is **not** a substitute for the authoritative S3 campaign.

### Run the production campaign on Kaggle GPU

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

Inspect the commands, remove `--dry-run`, and allow the tuned frozen counts to feed the S3 campaign.

## Submission strategy

The daily submission limit is useful as an **experiment budget**, not permission to fire ten nearly identical vectors at the leaderboard.

For the first cycle:

1. submit the primary S1-derived dual-view rank blend;
2. submit the mature combined-only control;
3. optionally submit one nearby blend weight if the first two provide useful separation;
4. preserve the remaining slots for S3/public-stack hypotheses.

Public leaderboard feedback is treated as weak evidence because the public subset introduces sampling noise. OOF remains the source of truth.

## Endgame

After S3, Frontier competes directly against the aligned public OOF universe. Public and Frontier streams are correlation-pruned together before bagged Caruana selection.

The key diagnostic is:

```text
frontier_total_weight
```

- **≥15%**: meaningful orthogonal signal
- **5–15%**: marginal contribution; focus on diversity
- **<5%**: stop tuning this backbone; it lies inside the public solution basin

## Nomophobia context

The repository name is inspired by the term **NOMOPHOBIA**, “NO MObile PHone PhoBIA.” Bhattacharya et al. discuss anxiety and distress associated with being detached from mobile-phone connectivity and emphasize the condition's overlap with other psychological disorders.

Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6510111/

A longer discussion of the relationship between the paper and this Kaggle task is in [`docs/SCIENTIFIC_CONTEXT.md`](docs/SCIENTIFIC_CONTEXT.md).

## Reproducibility status

- audit fixes A1/A4/A5/A7: implemented
- 20/20 v3.3 automated tests: passing in the development environment
- full 296,302-row test inference path: verified
- full S3 promotion: **pending Kaggle GPU campaign**
- public 74-stream pricing: **pending external OOF input**

## License

MIT. See [`LICENSE`](LICENSE).
