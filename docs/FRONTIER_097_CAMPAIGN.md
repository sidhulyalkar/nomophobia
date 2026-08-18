# NOMOPHOBIA 0.97+ Frontier Campaign

This document defines the next competition campaign after the live target-encoding frontier reached **0.967893241 honest five-fold OOF ROC AUC**.

The objective is not to tune the existing four-stream TE blend more finely. Its members have rank correlations around 0.997, so additional weight search is expected to spend validation signal rather than create useful information. The next campaign treats the current frontier as an anchor and asks whether deliberately different, target-free model contrasts can supply stable ranking corrections.

## Why 0.97+ is a credible leaderboard target

Public S6E8 work has already reported public scores above 0.9709. The strongest progression did not come from a single dramatically better standalone model. It came from a strong aligned anchor plus small residual directions from exact-value CatBoost, numeric identity/digit views, screen-relation features, model-family contrasts, fixed-schedule lookup/neural members, and target-free orthogonalization.

That evidence changes the optimization problem:

1. **Cross 0.97 by improving the anchor and its error diversity, not by over-tuning one booster.**
2. **Prefer matched control/treatment contrasts.** A treatment can be useful even when its standalone AUC is modest if the change it induces corrects anchor ordering errors.
3. **Select the first stable residual weight, not the best point on a dense grid.**
4. **Keep outer validation out of checkpoint selection.** Residual lanes use fixed schedules.
5. **Keep strict evidence separate from public-provenance diagnostics.** A useful external prediction library is not automatically a reproducible model claim.

No specific public score is guaranteed. The 0.97 threshold is realistic because it has been crossed publicly, while our current self-contained TE OOF score is still in the strong standalone-tree basin rather than the mature ensemble basin.

## New implementation

### `scripts/build_live_frontier_anchor.py`

Materializes the four-stream frontier as both an aligned OOF anchor and a validated test submission. The OOF side uses the already-recorded held-fold rotation weights; the test side uses the frozen mean weights. This closes an important operational gap in the older `build_live_frontier_blend.py`, which only emitted the test submission.

### `src/s6e8/identity.py`

Adds target-free representation views for matched ablations:

- exact float64 round-trip keys for CatBoost;
- float exponent and low mantissa bits;
- rounding at 0, 1, 2, and 3 decimals;
- residual-to-rounded-value features;
- decimal digit identities;
- screen allocation residuals and ratios.

The important object is not "more features" by itself. It is the **rank difference between an enhanced model and its matched raw control**.

### `src/s6e8/contrast.py`

Adds a prospective residual admission system:

- rank-space control/treatment directions;
- target-free direction orthogonalization;
- OOF-fitted/test-applied orthogonalization;
- first-passing weight selection over a predeclared grid;
- overall, per-fold, and even/odd-ID stability gates;
- rotating held-fold weight selection for honest OOF evaluation.

A candidate that cannot survive the gate simply deploys the unchanged anchor.

### `experiments/frontier_contrast_campaign.py`

Runs a matched fixed-schedule experiment on the exact outer folds of an existing anchor. It trains raw control and treatment models with identical seeds, folds, learner settings, and tree counts, then prices only their rank contrast against the anchor.

Default schedules:

| family | fixed rounds |
|---|---:|
| LightGBM | 900 |
| XGBoost | 1500 |
| CatBoost | 4000 |

There is deliberately **no outer-fold early stopping**. Raw-control predictions can be reused between treatments with `--reuse-control-dir` after the script verifies family, rounds, row count, and fold identity.

### `experiments/frontier_direction_composer.py`

Combines only directions that have already earned further testing. `equal_standardized` averages OOF-standardized directions; `sequential_orthogonal` first removes target-free linear overlap with earlier directions. Projection coefficients and normalization statistics are learned from OOF covariate geometry and frozen before application to test predictions.

## Execution order

### Phase 0: spend leaderboard slots on information

Submit the already-built controls before adding complexity:

1. four-stream `quad_v2` TE frontier;
2. inner-10 LGB/XGB control;
3. v0.1 frequency/raw dual-view control.

This estimates how the local TE basin transfers to the public leaderboard. Do not submit nearby 1-2% blend nudges.

### Phase 0.5: materialize the aligned quad anchor

The residual campaign must use the same four-stream frontier on train and test. Build it once from the completed full-data lanes:

```bash
python scripts/build_live_frontier_anchor.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --oof-s10-i5 artifacts/live_frontier_i5/oof.csv \
  --oof-s10-i10 artifacts/live_frontier_i10/oof.csv \
  --oof-s20-i5 artifacts/live_s20/oof_s20.csv \
  --test-lgb-s10-i5 artifacts/live_frontier_i5/submission_lgb.csv \
  --test-xgb-s10-i5 artifacts/live_frontier_i5/submission_xgb.csv \
  --test-lgb-s10-i10 artifacts/live_frontier_i10/submission_lgb.csv \
  --test-lgb-s20-i5 artifacts/live_s20/submission_s20.csv \
  --out-dir artifacts/frontier_quad_anchor
```

The next experiments then use:

```text
--anchor-oof  artifacts/frontier_quad_anchor/oof_anchor.csv
--anchor-test artifacts/frontier_quad_anchor/submission_anchor.csv
```

The builder reports both honest rotating OOF AUC and fixed-mean OOF AUC. They should reproduce the recorded frontier values to numerical tolerance before any residual experiment is trusted.

### Phase 1: matched LightGBM representation contrast

```bash
python experiments/frontier_contrast_campaign.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --anchor-oof artifacts/frontier_quad_anchor/oof_anchor.csv \
  --anchor-test artifacts/frontier_quad_anchor/submission_anchor.csv \
  --family lgb \
  --treatment identity_screen \
  --rounds 900 \
  --out-dir artifacts/contrast_lgb_identity_screen
```

If the combined treatment advances, split it into `identity` and `screen` only to learn which direction contributes useful geometry. If it fails, do not immediately increase trees.

### Phase 2: XGBoost matched contrast

```bash
python experiments/frontier_contrast_campaign.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --anchor-oof artifacts/frontier_quad_anchor/oof_anchor.csv \
  --anchor-test artifacts/frontier_quad_anchor/submission_anchor.csv \
  --family xgb \
  --treatment identity_screen \
  --rounds 1500 \
  --out-dir artifacts/contrast_xgb_identity_screen
```

Preserve the matched direction even if the standalone treatment does not win. The correction it induces can be more useful than the model itself.

### Phase 3: exact-value CatBoost contrast

```bash
python experiments/frontier_contrast_campaign.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --anchor-oof artifacts/frontier_quad_anchor/oof_anchor.csv \
  --anchor-test artifacts/frontier_quad_anchor/submission_anchor.csv \
  --family cat \
  --treatment identity_screen \
  --rounds 4000 \
  --device gpu \
  --out-dir artifacts/contrast_cat_identity_screen
```

CatBoost receives exact binary64 category keys in the treatment view. LightGBM and XGBoost do not, which keeps the high-cardinality identity experiment targeted to the learner most naturally suited to it.

### Phase 4: compose only accepted directions

```bash
python experiments/frontier_direction_composer.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --anchor-oof artifacts/frontier_quad_anchor/oof_anchor.csv \
  --anchor-test artifacts/frontier_quad_anchor/submission_anchor.csv \
  --direction-dirs \
    artifacts/contrast_lgb_identity_screen \
    artifacts/contrast_xgb_identity_screen \
    artifacts/contrast_cat_identity_screen \
  --mode sequential_orthogonal \
  --out-dir artifacts/contrast_composite
```

Do not feed every attempted direction into the composer. A failed direction is evidence, not ensemble inventory.

### Phase 5: price against an aligned public OOF universe

The repository already has `stack_external.py`. Once the self-contained directions are stable, use aligned public OOF predictions to answer a different question: whether our frontier contributes genuine orthogonal signal to the mature public ensemble basin.

Use fold provenance whenever available and prefer `fold_rank` plus bagged Caruana. If external arrays lack IDs/folds or model-selection provenance, label the resulting submission **diagnostic** rather than strict.

### Phase 6: neural diversity only if the residual budget still justifies it

A fixed-schedule Lookup Transformer or RealMLP is a sensible next diversity family after the tree contrasts. It is not the first move because it is more expensive and public evidence shows that OOF improvements can still transfer negatively to the leaderboard. Freeze epochs in advance and never choose a checkpoint on the outer fold.

## Prospective residual gate

Default gate for a five-fold campaign:

- overall AUC gain at least `1e-6`;
- at least 4/5 positive fold deltas;
- worst fold no worse than `-2e-6`;
- even- and odd-ID slices no worse than `-2e-6`;
- choose the **first** passing point on the declared weight grid;
- repeat selection with each outer fold held out;
- deployment weight is the median rotating weight;
- if the honest assembled OOF candidate fails the final gate, deployment weight becomes zero.

These thresholds are intentionally permissive enough for endgame micro-residuals but much harder to game than maximizing one full-OOF weight grid.

## Promotion ladder

| level | requirement | action |
|---|---|---|
| S1 direction screen | matched contrast, positive overall and majority folds | keep direction artifacts |
| S2 residual gate | rotating held-fold gate passes | materialize candidate submission |
| LB contrast | candidate beats/equals refreshed incumbent within expected noise | continue family/composite |
| S3 | replicated seeds or independent fixed schedules, strict lineage | promote to repo winner |

A leaderboard improvement never retroactively sanitizes a leaky validation path. Conversely, a strict OOF micro-gain that loses publicly should be preserved as negative-transfer evidence.

## Decision rules

- If `quad_v2` is already >= 0.9700 publicly, shift immediately to conservative micro-residuals and S3 robustness.
- If the TE controls cluster around ~0.969, prioritize the residual lab and external-library anchor before more TE smoothing/capacity work.
- If all three matched tree contrasts receive zero deploy weight, stop engineering adjacent tree features and move to Lookup/RealMLP or a richer aligned public OOF ensemble.
- If a direction helps standalone treatment AUC but not the anchor residual, kill it as an ensemble direction.
- If a direction hurts standalone AUC but passes the residual gate, keep it. Diversity is the product.

## Immediate target

The near-term target is a **strict, reproducible 0.97+ leaderboard submission**, not a claim of competition-winning status. After 0.97 is crossed, the next goal is to determine whether the remaining public frontier can be reproduced from auditable components rather than accumulated from opaque prediction artifacts.
