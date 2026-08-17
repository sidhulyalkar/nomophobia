# NOMOPHOBIA 📵

**NOMOPHOBIA: NO MObile PHone PhoBIA**

A falsification-first research system for Kaggle Playground Series Season 6 Episode 8, **Predicting Smartphone Addiction**.

> Scientific note: the project name references the nomophobia literature, but this repository predicts Kaggle's synthetic binary `addicted_label`; it is not a clinical diagnostic system.

## Current competition thesis

The strongest corrected evidence points to a **population-frequency + behavioral-decomposition** problem, with a deliberately raw model supplying potentially useful error diversity.

```text
                         row
                          │
              ┌───────────┴───────────┐
              │                       │
       engineered view             raw view
              │                       │
 behavior + missingness          12 predictors
 + population frequency               │
 + small digit block                  │
              │                       │
       LGBM combined63            LGBM raw63
              │                       │
              └───────────┬───────────┘
                          │
                     rank blend
```

Existing controlled ablations attribute roughly **+0.0038 AUC** to the frequency block, **+0.0018** to behavioral decomposition, and only **+0.0002** to the entire digit/rounding family. Five independent S1 dual-view replications were positive: three at 700 trees and two at 1000 trees. This is strong screening evidence, not an S3 promotion claim.

The current v0.1 submission remains frozen at:

```text
combined63: 1000 trees
raw63:      1000 trees
score = 0.625 × rank(combined) + 0.375 × rank(raw)
```

The first full-data test inference produced higher raw/combined rank correlation than the S1 screens, so v0.3 treats **capacity-dependent diversity collapse** as a first-class hypothesis instead of assuming the 37.5% raw weight will survive S3.

## What v0.3 changes

v0.3 turns the research codebase into an evidence-driven campaign engine while preserving the current winner until new ideas earn promotion.

### Installable CLI

```bash
pip install -e ".[dev]"
nomophobia --help
```

Commands:

```text
nomophobia validate   validate competition files and provenance
nomophobia route      inspect completed artifacts and recommend the next experiment
nomophobia tune       repeated production-scale iteration calibration
nomophobia s3         authoritative 3-seed × 5-fold campaign
nomophobia winner     tune both experts, run S3, route the result, materialize submission
```

The historical `run_winner.py`, `run_s3.py`, and `tune_iterations.py` entrypoints remain as compatibility wrappers.

### Universal experiment manifests

Every new campaign can emit a failure-safe manifest containing:

- evidence tier;
- hypothesis and falsifying measurement;
- accept and kill rules;
- Git SHA / branch when available;
- Python and package versions;
- competition CSV SHA-256 hashes;
- exact configuration and seeds;
- elapsed time;
- metrics;
- output file hashes;
- crash status and traceback when a run fails.

See [`docs/EXPERIMENT_MANIFEST.md`](docs/EXPERIMENT_MANIFEST.md).

### Repeated production-scale tuning

A single inner holdout can choose a noisy tree count. v0.3 tunes each LightGBM expert on repeated production-scale holdouts and freezes the **median** best iteration for all S3 folds.

```bash
nomophobia tune \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --expert lgb_combined63 \
  --rows 628000 \
  --max-estimators 4000 \
  --patience 200 \
  --repeats 3 \
  --device gpu
```

The job stops for inspection if any repeat hits the estimator ceiling or if the selected iteration range exceeds 30% of its median.

## Frequency research program

### 1. Mature source-safety stress

```bash
python experiments/frequency_stress.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --rows 120000 \
  --estimators 1000 \
  --source-rows 100000 \
  --device gpu \
  --out artifacts/frequency_stress.json
```

This now reports three different source tests:

1. frequency-only train vs test AUC on all sampled rows;
2. missingness-only train vs test AUC;
3. **frequency-only train vs test AUC on complete rows**.

The third is the important kill switch. A source signal explained by missingness is different from a hidden fingerprint among otherwise complete records.

### 2. Marginal frequency-family decomposition

```bash
python experiments/frequency_family_ablation.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --rows 120000 \
  --estimators 1000 \
  --device gpu \
  --out artifacts/frequency_family_ablation.json
```

Arms are `full`, `none`, `exact_only`, `rounded_only`, `categorical_only`, and `exact_plus_rounded`. Reduced arms are now compared to `full` with paired OOF confidence intervals rather than naked point AUCs.

### 3. Higher-order target-free density geometry

```bash
python experiments/frequency_geometry.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --rows 120000 \
  --estimators 1000 \
  --device gpu \
  --out-dir artifacts/frequency_geometry
```

The experiment tests:

- selected pair frequencies such as daily×social and daily×weekend;
- selected triple frequencies such as daily×social×weekend;
- smoothed interaction / conditional-density quantities;
- value frequency conditioned on the row's **other-feature missingness regime**;
- unsigned train/test support stability.

No target means are used. Source-stability arms have their own complete-row source-adversarial stop gate.

### 4. Capacity-dependent diversity curve

```bash
python experiments/capacity_diversity_curve.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --rows 180000 \
  --estimators 400 700 1000 1400 2000 \
  --device gpu \
  --out artifacts/capacity_diversity_curve.json
```

This deliberately fixes the raw weight at 37.5%. It asks whether raw/combined rank correlation rises toward the production warning region as capacity increases and whether a fixed raw hedge still changes pairwise ordering beneficially. It is a routing experiment, not another blend-weight optimizer.

## A new source-data hypothesis

Earlier source experiments tried source severity as a feature, blend member, or boosting prior. v0.3 adds a different test: **direct low-weight labeled-row augmentation**.

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

Exact predictor-overlap source rows are removed before training. Source rows are never used as validation evidence. The fixed broad weight grid is intentional: a result that exists only at a finely optimized source weight is not interesting enough to promote.

## Authoritative S3

The main campaign is now:

```bash
nomophobia winner \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --out-root artifacts/winner \
  --device gpu \
  --tune-rows 628000 \
  --max-estimators 4000 \
  --patience 200 \
  --tune-repeats 3
```

S3 still means **3 distinct seeds × 5 frozen folds × all 691,369 labeled rows**. Promotion requires:

- at least 13/15 positive fold-seed deltas;
- pooled paired CI above zero;
- positive test-missingness-weighted delta;
- candidate fold-AUC standard deviation within 120% of baseline.

v0.3 additionally diagnoses:

- blend-AUC spread across seeds;
- raw/combined rank correlation at production capacity;
- optimistic-vs-honest blend selection gap;
- rotation-weight mean/std/range across all held folds.

If S3 promotes the dual view, the backbone is frozen and mature CatBoost/XGBoost/Evidence retrials become the next priority. If combined promotes but raw does not, the system stops probing tiny raw weights and redirects effort to genuinely new diversity sources.

When a model is promoted, the winner campaign copies the appropriate seed-bag candidate to `submission_s3.csv`, validates it against test IDs, and records its SHA-256.

## Evidence router

After experiments complete:

```bash
nomophobia route \
  --artifact-root artifacts \
  --data-dir /kaggle/input/playground-series-s6e8
```

The router inspects what evidence is actually present and emits the next highest-value command. The goal is to avoid the classic Kaggle research swamp where dozens of half-tested branches accumulate and nobody remembers which uncertainty matters most.

See [`docs/RESEARCH_ROADMAP_V03.md`](docs/RESEARCH_ROADMAP_V03.md) and [`docs/KAGGLE_RUNBOOK.md`](docs/KAGGLE_RUNBOOK.md).

## Production safety rails

`load_competition()` validates exact columns, ID uniqueness/order, target validity, and finite-or-missing numeric predictors. Submission generation validates row count and test-ID order, rejects non-finite/constant scores, writes atomically, re-reads from disk, validates again, and SHA-256 hashes the artifact.

## Research discipline

- S0: plumbing/direction only.
- S1: may advance, never promote.
- S2: effect-size confirmation.
- S3: only promotion tier.
- Every claim uses identical-row paired comparisons where applicable.
- Failed ideas remain in [`docs/DECISIONS.md`](docs/DECISIONS.md).
- A surprise >+0.003 improvement is treated as a leak alarm before a win.
- A zero optimized ensemble weight is never sufficient evidence to kill a diversity member.

## Endgame

After S3, the frozen Frontier streams are priced against the aligned public OOF universe using correlation pruning and bagged Caruana selection. `frontier_total_weight` is the key diagnostic:

- **≥15%**: meaningful orthogonal signal;
- **5–15%**: marginal; prioritize diversity;
- **<5%**: stop tuning the same LightGBM basin.

## Status

- v0.2 production contracts: implemented.
- v0.3 package CLI + manifests: implemented on the current development release.
- mature frequency geometry / source-row augmentation: implemented, evidence pending execution.
- authoritative S3: pending production GPU execution.
- competition-winning status: **not claimed until leaderboard and S3 evidence support it**.

## License

MIT. See [`LICENSE`](LICENSE).
