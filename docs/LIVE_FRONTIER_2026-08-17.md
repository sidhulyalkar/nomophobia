# Live TE frontier evidence — 2026-08-17 PDT / 2026-08-18 UTC

This note records the full-data target-encoding frontier that was executed after the v0.3.1 CPU/Kaggle infrastructure landed. It is intentionally separate from S3 promotion evidence: these are single-seed five-fold OOF comparisons plus replicated S1 screens, useful for leaderboard routing but not enough to claim a promoted competition winner.

## Competition contract

- Train rows: **691,369**
- Test rows: **296,302**
- Target: `addicted_label`
- Metric: ROC AUC
- Outer split: 5-fold stratified CV, seed `20260801`
- Numeric target encoding: nested exact-value posterior mean with smoothing
- Composition features: component sum, social/gaming/work-study shares, weekend-minus-daily
- LightGBM: learning rate 0.035, 31 leaves, early stopping 150
- XGBoost: histogram trees, depth 8, min child weight 20, early stopping 150
- Ensemble comparisons use percentile ranks.

## Replicated screens

### Encoding uncertainty metadata: KILL

`experiments/te_uncertainty_screen.py` compared plain exact-value TE against the same TE plus support log-count, posterior reliability, and evidence-z features on three matched 120k samples.

| seed | baseline AUC | uncertainty AUC | delta |
|---:|---:|---:|---:|
| 20260816 | 0.963978605 | 0.963529165 | -0.000449441 |
| 20260817 | 0.964828183 | 0.964595927 | -0.000232256 |
| 20260818 | 0.965176376 | 0.965192379 | +0.000016003 |

Mean delta: **-0.000221898**. Only 1/3 seeds was positive.

Interpretation: the posterior mean itself is useful; appending explicit confidence metadata did not improve the tree model.

### Inner encoding folds: keep inner-5 as the default standalone setting

`experiments/te_innerfold_screen.py` compared 5 versus 10 nested encoding folds on three matched 120k samples.

| seed | inner-5 AUC | inner-10 AUC | delta |
|---:|---:|---:|---:|
| 20260816 | 0.963978605 | 0.963905639 | -0.000072966 |
| 20260817 | 0.964828183 | 0.964714850 | -0.000113333 |
| 20260818 | 0.965176376 | 0.965155731 | -0.000020644 |

Mean delta: **-0.000068981**. Inner-10 lost on all three seeds at S1 scale.

This does not make inner-10 useless. At full scale it became a slightly better ensemble partner with XGBoost, which is why the live blend retains it as a separate stream.

### Small-screen capacity: plateau before 1,200 trees

`experiments/te_capacity_screen.py` trained one 2,000-tree booster per seed and evaluated the same held-out rows at 600, 1,200, and 2,000 trees.

Mean 1,200 minus 600: **-0.000090024**.
Mean 2,000 minus 1,200: **-0.000351913**.
All three seeds worsened from 1,200 to 2,000.

Interpretation: tree-count optima scale strongly with training-set size. Small S1 screens must not be used to freeze the 691k production tree count. Full-data lanes therefore use early stopping.

### Smoothing diversity: ADVANCE

`experiments/te_smoothing_diversity.py` compared smoothing 10, smoothing 20, and their fixed 50/50 rank blend on three matched 120k samples.

| seed | s10 AUC | s20 AUC | 50/50 blend AUC | blend gain over best |
|---:|---:|---:|---:|---:|
| 20260816 | 0.963978605 | 0.964070604 | 0.964254101 | +0.000183497 |
| 20260817 | 0.964663412 | 0.964658631 | 0.964886176 | +0.000222764 |
| 20260818 | 0.965071435 | 0.964880495 | 0.965218225 | +0.000146790 |

Mean blend gain: **+0.000184350**, positive on 3/3 seeds.

This was the only new feature-side TE result in this batch that replicated positively enough to justify a full-data lane.

## Full-data five-fold lanes

### Smoothing-10, inner-5 LightGBM + XGBoost

`experiments/live_frontier_candidate.py --inner-folds 5 --smoothing 10`

- LightGBM OOF AUC: **0.967581878**
- XGBoost OOF AUC: **0.967688247**
- LGB/XGB rank correlation: **0.997184537**
- Rotating held-fold LGB weights: `[0.40, 0.45, 0.45, 0.45, 0.45]`
- Mean LGB weight: **0.44**
- Honest rotating-fold blend AUC: **0.967825597**

XGBoost best iterations by fold: 1405, 1411, 1390, 1426, 1520.
LightGBM best iterations by fold: 3564, 2628, 3276, 2498, 2622.

### Smoothing-10, inner-10 LightGBM

The full inner-10 LightGBM lane produced:

- OOF AUC: **0.967644407**
- Best iterations by fold: 3032, 3099, 3288, 2580, 3428

It is better than the inner-5 LightGBM standalone on this particular full run, despite losing the replicated 120k standalone screen. The important use is as a distinct blend stream rather than as proof that 10 inner folds are universally superior.

When paired with the same full XGBoost stream, rotating held-fold selection chose LGB weights `[0.50, 0.45, 0.45, 0.45, 0.50]`, mean **0.46**, for honest OOF AUC **0.967867071**.

### Smoothing-20, inner-5 LightGBM

`experiments/live_smoothing20_candidate.py`

- OOF AUC: **0.967570470**
- Best iterations by fold: 3457, 3378, 3370, 2754, 3157

Smoothing-20 is not stronger standalone at full scale. Its value is diversity.

## Four-stream live blend

Aligned OOF streams were priced together:

1. smoothing-10 inner-5 LightGBM
2. smoothing-10 inner-5 XGBoost
3. smoothing-10 inner-10 LightGBM
4. smoothing-20 inner-5 LightGBM

Rank correlations are all high, roughly 0.9967 to 0.9982, so this is a narrow frontier rather than a new model family.

A coarse rotating held-fold selection over nearby simplex weights chose:

| held fold | s10/i5 LGB | XGB | s10/i10 LGB | s20/i5 LGB |
|---:|---:|---:|---:|---:|
| 0 | 0.125 | 0.45 | 0.30 | 0.125 |
| 1 | 0.15 | 0.45 | 0.25 | 0.15 |
| 2 | 0.15 | 0.45 | 0.25 | 0.15 |
| 3 | 0.175 | 0.40 | 0.25 | 0.175 |
| 4 | 0.15 | 0.45 | 0.25 | 0.15 |

Mean weights are therefore:

```text
0.15 × rank(s10 inner-5 LGB)
0.44 × rank(s10 inner-5 XGB)
0.26 × rank(s10 inner-10 LGB)
0.15 × rank(s20 inner-5 LGB)
```

- Honest rotating-fold OOF AUC: **0.967893241**
- Fixed mean-weight OOF AUC: **0.967894911**
- Gain vs inner-10/XGB honest blend: about **+0.000026**
- Gain vs original inner-5/XGB honest blend: about **+0.000068**

The effect is small. Treat this as a leaderboard probe, not a promotion claim.

## Submission artifacts

The locally materialized four-stream file is:

```text
submission_nomophobia_frontier_quad_v2.csv
```

Validated contract:

- rows: 296,302
- columns: `id`, `addicted_label`
- IDs aligned to test order
- no duplicate IDs
- no non-finite predictions
- continuous ranking scores
- SHA-256: `3d714f3ab11584e1df9066afea0875ebba689513dd94c018be4b47c950740ede`

The simpler inner-10/XGB candidate remains useful as a control:

```text
submission_nomophobia_frontier_i10_xgb_v1.csv
```

- weights: 0.46 inner-10 LGB / 0.54 XGB
- honest OOF AUC: 0.967867071
- SHA-256: `693119723419d50bee92276e827d87d7d86768d0e9840bd73bf20c605e0a1f49`

## Reproduction

Run the primary full frontier:

```bash
python experiments/live_frontier_candidate.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --out-dir artifacts/live_frontier_i5 \
  --folds 5 --inner-folds 5 --smoothing 10 \
  --lgb-estimators 4500 --xgb-estimators 3000
```

Run the inner-10 lane with the same script by changing `--inner-folds 10`.

Run smoothing-20:

```bash
python experiments/live_smoothing20_candidate.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --out-dir artifacts/live_s20 \
  --estimators 4500
```

Materialize the four-stream submission from the saved test CSVs:

```bash
python scripts/build_live_frontier_blend.py \
  --lgb-s10-i5 artifacts/live_frontier_i5/submission_lgb.csv \
  --xgb-s10-i5 artifacts/live_frontier_i5/submission_xgb.csv \
  --lgb-s10-i10 artifacts/live_frontier_i10/submission_lgb.csv \
  --lgb-s20-i5 artifacts/live_s20/submission_s20.csv \
  --out submission_nomophobia_frontier_quad_v2.csv
```

The manual GitHub Actions workflow `.github/workflows/live_frontier_manual.yml` can run each research lane explicitly via `workflow_dispatch`. It uses a public mirror only as CI convenience; accepted Kaggle competition files remain the source of truth for production submissions.

## Next decision

The highest-value next external measurement is the public leaderboard contrast between:

1. the four-stream `quad_v2` candidate;
2. the simpler inner-10/XGB control;
3. the existing v0.1 frequency + raw dual-view submission.

Do not spend slots on nearby 1–2% weight nudges unless the leaderboard establishes that this TE basin transfers cleanly.
