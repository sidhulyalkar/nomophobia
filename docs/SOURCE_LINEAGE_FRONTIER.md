# Source-Lineage Frontier

## Why this lane exists

The current campaign has reached the point where stronger versions of the same representation mostly collapse into the same ranking basis. The audited public v19 anchor is already around 0.9701 OOF, while the replicated LightGBM identity/screen direction contributes only a few millionths of AUC once priced against that mature anchor.

The next experiment therefore targets a different object: **how the competition generator reused the 7,500-row original population**.

This is deliberately not another nearest-source model, source-severity prior, or low-weight source-row augmentation. Those hypotheses are already represented in the decision log and did not produce a robust global improvement.

## Structural hypothesis

The original public dataset contains the same 12 predictors plus `addiction_level` and `addicted_label`. Synthetic tabular generators often preserve some values or small feature subsets from their source population even when the complete generated row is novel.

For a competition row and a subset such as:

```text
[daily_screen_time_hours, social_media_hours, weekend_screen_time]
```

we ask whether that exact normalized tuple appears in any original row.

Across every 1-, 2-, and 3-feature subset, the encoder first measures this membership coverage on a target-free sample of competition train+test rows. It then retains a bounded set of non-degenerate subset signatures. **Competition labels are never used to choose the subsets.**

This creates two matched treatments:

1. `membership`: exact subset membership and source recurrence counts only. This is a pure generator-lineage test.
2. `lineage`: the same membership representation plus smoothed local source `addicted_label` log-odds and local four-level severity support. This tests whether the copied support also carries useful latent-class information.

The matched control is the exact same fixed-900 LightGBM raw model already measured by the frontier campaign.

## Why this is different from killed source experiments

### G4 nearest-original severity

G4 asked which original row is closest globally and transferred the nearest source severity/distance. It was strongly predictive standalone but redundant or harmful once added to the competition backbone.

The lineage experiment does not assign one global nearest parent. It asks which **partial source signatures survive exactly**, allowing a synthetic row to be a mosaic of multiple source records or generator states.

### G5 recurring competition keys

G5 showed that partial keys recur inside the competition data, but fold-safe empirical-Bayes target means on those competition keys were harmful.

The lineage experiment uses **membership in the external original population**, not recurrence inside competition train, and its subset selection is target-free. The model learns whether source-membership itself changes the competition target distribution.

### Source ordinal prior / source augmentation

A smooth source classifier compresses the source population into a global prediction surface. Low-weight augmentation asks the competition learner to absorb 7,500 source rows among hundreds of thousands of synthetic rows.

Lineage instead preserves discrete provenance. A row can have a weak smooth source prior while still exposing a highly informative exact source-subset fingerprint.

## Validation contract

The experiment inherits the frontier campaign's fixed outer folds and fixed 900-tree schedule. Outer validation is never used for checkpoint selection.

For each treatment:

```text
raw control prediction
        │
        ├──────────────┐
        ▼              ▼
   raw features   raw + lineage features
        │              │
        └──── rank(treatment) - rank(control)
                         │
                         ▼
                 residual vs anchor
```

The residual must pass rotating held-fold selection with zero available in the weight grid. It is also stress-tested across:

- source-lineage reliability bands;
- low / boundary / high anchor-score regions;
- zero, one, and multiple missing-feature regimes;
- the existing fold and even/odd-ID gates.

If the structural stress gate fails, deployment weight is forced to zero even when pooled OOF rises.

## Full-data execution

The workflow is:

```text
.github/workflows/source_lineage_compute.yml
```

It reuses the validated aligned quad anchor and the matched raw LightGBM control from full-data run `32194910518`, so the new compute is spent only on the new treatment models.

It runs both treatments in parallel:

```text
membership
lineage
```

Each produces:

```text
results/
├── decision.json
├── source_lineage_screen.json
├── oof.csv
├── direction_oof.npy
├── direction_test.npy
├── submission_diagnostic.csv
└── submission_gated.csv
```

`submission_gated.csv` is fail-closed: when the residual or structural gate fails, it is the unchanged anchor.

## Decision ladder

### If membership wins and lineage does not

The generator-copy fingerprint is real, while direct source-label transfer is redundant. Expand the provenance representation rather than the source predictor:

- order-4 subset screening;
- source-parent coherence / maximum mutually consistent subset order;
- orthogonalize the membership direction against identity/screen before composition.

### If lineage wins beyond membership

The original four-level severity boundary survives locally in the generator. Next build a **boundary specialist** that targets source-supported rows near the Mild/Moderate transition instead of applying source evidence globally.

### If both fail

Kill source provenance as a global residual family. The next high-value basis should be a generator-likelihood or fixed-partition neural residual, not another tree-feature permutation.

### If either passes strongly

Price the accepted direction against the audited v19 OOF anchor, not only the self-contained quad anchor. Only after it survives that mature-anchor head-to-head should it replace `kaggle/submission_nomophobia_v19_lgb015.csv`.

## External precedent

A 2nd-place solution in Playground Series S4E1 exploited syntheticity by generating subset-membership indicators that recorded whether feature subsets appeared unchanged in the original dataset. That solution is precedent for the mechanism, not evidence that it works here. This campaign is designed to test the mechanism honestly on S6E8.

Reference: <https://www.kaggle.com/competitions/playground-series-s4e1/discussion/472496>

Original smartphone source mirror used by the workflow: <https://github.com/Chireal-218/Smartphone-Usage-and-Addiction-Analysis>
