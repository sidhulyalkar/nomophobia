# Winning strategy — 2026-08-19

## Executive thesis

NOMOPHOBIA is no longer bottlenecked by access to stronger endpoint classifiers. The mature audited v19 blend plus the accepted 0.015 LightGBM identity/screen residual reaches 0.9701014002 OOF, and the public leaderboard response curve shows that increasing ownership of the same LightGBM family degrades transfer.

The remaining frontier is **error modeling**: identify systematic pair-ordering mistakes made by the mature champion, learn only the residual needed to repair those mistakes, and deploy corrections in regions where they remain stable under held-fold and structural gates.

The first successful example is the pairwise low-tail specialist in PR #23.

## What we now understand about the problem

The competition has at least three useful layers of signal.

### 1. Behavioral geometry

Daily screen time, social-media time, weekend screen time, and their allocation/interaction geometry dominate the underlying addiction axis. The 7,500-row public source dataset is especially close to a daily-screen/social-media boundary.

This signal is important but no longer scarce. Mature public blends and all strong tree models already capture most of it.

### 2. Synthetic-generator fingerprints

Exact numeric identity, screen-allocation features, target encoding, value frequency, and source-subset membership all reveal structure introduced or preserved by the synthetic generator.

The important result is not that these features are weak. They are strong. The matched LightGBM and XGBoost identity/screen treatments gain about +0.0024 standalone, and source-label-aware lineage adds about +0.00137 to a raw LightGBM control.

The problem is redundancy: after pricing them against the mature champion, almost all of that gain disappears.

### 3. Residual ordering mistakes

The mature champion still makes systematic ROC-AUC ordering errors. Those errors are not distributed uniformly.

The free-standing pairwise ranker failed globally, but its correction became useful when restricted to the low-score tail. Under a fully nested cutoff/sign/magnitude selector, every held fold chose a specialist acting only on the bottom 30% of champion ranks.

That is the first new basis in this campaign that survives the exact mature champion rather than a weaker proxy.

## Evidence ledger

### Keep

#### v19 + LightGBM identity/screen residual

Incumbent recipe:

```text
rank(rank(v19) + 0.015 * lgb_identity_screen_direction)
```

OOF AUC: `0.9701014001897124`.

This remains the production anchor until a challenger transfers on the leaderboard.

#### Pairwise low-tail specialist

Accepted challenger:

```text
anchor = v19 + LGB015
mask = rank(anchor) <= 0.30
candidate = rank(rank(anchor) - 0.0025 * pairwise_orthogonal_direction * mask)
```

Nested honest OOF:

- AUC: `0.9701033965102790`
- gain: `+0.0000019963205666`
- fold wins: `5/5`
- even-ID delta: `+0.00000273195`
- odd-ID delta: `+0.00000119181`

Fixed deployment OOF:

- AUC: `0.9701036395944417`
- gain: `+0.0000022394047293`
- fold wins: `5/5`
- worst fold delta: `+0.00000062364`
- structural gate: `PASS`

This is a leaderboard-worthy challenger, not yet a production replacement.

### Kill as global residual families

#### More LightGBM ownership

The public response curve degrades as the submission moves farther from the mature anchor toward the identity/screen LightGBM endpoint. Do not spend submissions sweeping larger LGB weights.

#### Source provenance

Exact source-subset membership is real signal, but its held-fold residual weight against the frontier anchor is zero. Adding source labels and four-level severity improves raw LightGBM by about +0.00137 but contributes essentially zero honest residual against the mature ensemble.

Keep source information only as a possible specialist/gating variable, not a global endpoint.

#### Generative TAN / Naive Bayes

Tree-Augmented Naive Bayes is genuinely diverse and improves Naive Bayes by about +0.0237 standalone, with only about 0.943 rank correlation to the frontier anchor. Nevertheless every nonzero TAN/Naive residual probe hurts the anchor.

Diversity alone is not enough.

#### Blind public-stack expansion

The audited 74-stream library does not improve merely by averaging or simple rank-logit stacking. Future use of that library should model **champion error**, not form another endpoint blend.

## Highest-value next experiments

### A. Champion-offset RankNet

Instead of training a new score and blending afterward, fit pairwise residual coefficients with the champion's positive/negative margin frozen as an offset:

```text
softplus(-(T * champion_margin + residual_feature_margin))
```

Mix uniform positive/negative pairs with hard pairs across broad champion-rank separations. This directly optimizes what feature evidence remains useful after the champion.

This is implemented in stacked draft PR #24.

### B. Test-distribution-aware residual validation

Train a target-free domain classifier to distinguish train rows from test rows using raw features, missingness, identity/frequency features, and generator signatures.

Use the resulting density ratio only to **reweight OOF residual evaluation**, not to read test labels. The objective is to answer:

> Which tiny residuals improve the rows that look most like the hidden test distribution?

This is especially valuable because public work reports measurable train/test missingness shift and because our leaderboard response curve is already precise enough to expose CV-to-LB transfer error.

### C. Champion-error meta-model over the public OOF library

The 74 public streams should be treated as explanatory variables for champion error, not as equal peers in another stack.

For each row, use target-free disagreements such as:

```text
rank(stream_j) - rank(champion)
```

and train a cross-fitted pairwise or conditional residual model. Strong correlated streams can still be useful if their disagreement predicts a specific champion mistake.

### D. Region-specific specialists

The low-tail result proves that global residual gates can hide useful conditional signal. Future specialists must be predeclared and nested:

- low-score tail;
- high uncertainty / high model disagreement;
- test-like domain propensity;
- high source-support reliability;
- missingness regimes.

A specialist is promoted only if its region rule, direction, and coefficient are selected without scoring the held outer fold.

## Submission policy

Leaderboard slots are an external sensor, not the CV loop.

The recommended order is:

1. retain `submission_nomophobia_v19_lgb015.csv` as the incumbent reference;
2. submit the accepted pairwise low-tail specialist once;
3. keep the XGBoost residual probe as a diagnostic if it has not yet been measured;
4. do not submit source-lineage, TAN, larger-LGB, or arbitrary blend-weight variants;
5. reserve remaining slots for residual families that beat the exact mature champion under nested OOF.

A leaderboard gain is then used to estimate transfer of a **mechanism**, not to tune another continuous coefficient sweep.

## Win condition

The current gap is small enough that the goal should not be a giant new standalone model. A competition win is more plausibly assembled from several orthogonal, individually tiny residuals that each survive exact mature-anchor validation and transfer externally.

The project should therefore optimize for:

```text
new basis quality × validation honesty × test transfer
```

not raw standalone AUC.
