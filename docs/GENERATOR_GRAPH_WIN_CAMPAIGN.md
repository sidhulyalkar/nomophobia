# Generator-Graph Win Campaign

## Why the current frontier is no longer a tuning problem

The nested frontier winner already demonstrates that the strongest local gains come from **small rank corrections around a mature anchor**, not from replacing the anchor with a nominally stronger single model.

Measured evidence on the current branch:

- aligned four-stream TE anchor honest OOF AUC: `0.9678852781`;
- nested LGB + XGB identity/screen correction honest OOF AUC: `0.9681718649`;
- honest gain: `+0.0002865868`;
- all five held folds improve;
- paired DeLong 95% gain interval is approximately `[+0.0002548, +0.0003184]`;
- current deployment weights are `0.20 × LGB direction + 0.10 × XGB direction`.

That is strong enough to establish a strategy change. Additional tree-family weight tuning is now low expected value. The missing upside must come from a new representation of the **synthetic data-generating process**.

## New hypothesis: latent source identity survives synthesis

The existing target-encoding frontier treats each exact numeric value independently. That can exploit generator memorization when a source value is repeated, but it discards a stronger possibility:

> A synthetic record may retain several fragments of the same latent source record at once.

If so, the important object is not a single repeated value. It is the **intersection pattern of repeated/rounded values across columns**.

The new generator graph models every row as connected to target-free identity tokens. The matched control contains only univariate exact-value tokens. The treatment adds predeclared joint tokens such as:

- daily screen time × social-media time;
- daily × weekend screen time;
- daily × work/study time;
- social × gaming;
- sleep × daily screen time;
- notifications × app opens;
- age × daily/weekend usage;
- stress × daily screen time;
- academic impact × work/study time;
- gender × social-media time.

Every token becomes a node in a bipartite row-token graph. Label evidence is attached only from the reference training rows. A query row receives an empirical-Bayes posterior from the token nodes it touches, weighted by effective support.

This creates a direct test of latent source reconstruction without requiring the original source dataset.

## Leakage contract

The graph is strictly outer-fold safe:

1. For held fold `k`, token posteriors are estimated from folds `!= k` only.
2. Held-fold labels never enter token statistics.
3. Test predictions are averaged from the same fold-trained graph models used to create OOF predictions.
4. The structural direction is `rank(joint_graph) - rank(exact_only_graph)`, so the residual prices only information beyond univariate exact-value TE.
5. Residual weight selection uses the repository's rotating held-fold gate.
6. If the gate fails, the emitted submission is exactly the unchanged frontier winner.

## What constitutes a breakthrough

The structural hypothesis is **promoted** only if all of these hold:

- joint graph beats exact-only graph OOF;
- the graph direction improves the current nested frontier anchor;
- rotating selection produces a non-zero median deployment weight;
- at least 4/5 folds improve and the worst fold remains inside the prospective tolerance;
- even/odd ID slices pass;
- the frozen deployment candidate remains positive against the frontier winner.

A large standalone graph AUC without a positive frontier residual is not useful. A small standalone graph AUC with a stable frontier correction is useful.

## Command

After reconstructing the current frontier winner anchor:

```bash
python experiments/generator_graph_frontier.py \
  --data-dir /tmp/s6e8 \
  --anchor-oof artifacts/frontier_winner/oof_frontier_winner.csv \
  --anchor-test artifacts/frontier_winner/submission_frontier_winner.csv \
  --anchor-oof-col frontier_winner \
  --smoothing 10 \
  --min-support 2 \
  --out-dir artifacts/generator_graph
```

Primary artifact:

```text
artifacts/generator_graph/submission_generator_graph.csv
```

The runner validates IDs/schema, saves aligned OOF/test graph arrays, records token coverage diagnostics, applies the rotating residual gate, validates the submission through the repository submission contract, and records its hash.

## Next structural escalations if v1 is positive

Do not immediately tune smoothing. First decompose the source of the gain.

1. **Leave-one-joint-family-out attribution** to identify which relationships contain generator identity.
2. **Three-way source tokens** only for families that already pass pairwise attribution.
3. **Reliability calibration by support and entropy**, learned only on non-held folds.
4. **Graph diffusion**: row → token → row → token propagation with labels injected only at reference rows.
5. **Nearest-source posterior** using collision-weighted Hamming distance across token families.
6. **Original-data linkage**, if a provenance-safe source dataset is available, as a separate experiment from synthetic-only graph evidence.

## Kill rule

If the exact-vs-joint graph direction receives zero deployment weight, stop expanding pair grids. That result would say the generator's useful memorization is already captured by marginal exact-value TE, and the next major lane should instead be representation diversity such as RealMLP/TabPFN-style distillation or a public-OOF meta-ensemble.

## Competition objective

The goal is not to claim a win from OOF. The goal is to create a sequence of hypotheses where each leaderboard submission answers a structural question. At this stage, the highest-value question is whether the synthetic generator leaks **latent source neighborhoods**, because that is one of the few mechanisms capable of moving materially beyond the current 0.97-class ensemble basin rather than merely reshuffling it.
