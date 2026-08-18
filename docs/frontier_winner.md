# Frontier residual winner

This document records the current evidence-backed submission candidate produced without using Kaggle compute or leaderboard feedback.

## Validated anchor

The full-data GitHub Actions campaign (run `32194910518`) reconstructed the aligned four-stream target-encoding anchor from:

- LightGBM, smoothing 10, inner 5
- XGBoost, smoothing 10, inner 5
- LightGBM, smoothing 10, inner 10
- LightGBM, smoothing 20, inner 5

The fresh honest OOF AUC is **0.9678852781**.

## Matched residual experiments

Fixed-schedule contrasts were trained on exactly the anchor's five outer folds. The outer validation fold was never used for early stopping or checkpoint selection.

| Family | Treatment | Fixed schedule | Control OOF | Treatment OOF | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| LightGBM | identity + screen relations | 900 trees | 0.9616915351 | 0.9640999877 | +0.0024084526 |
| XGBoost | identity + screen relations | 1500 trees | 0.9640292546 | 0.9664867754 | +0.0024575209 |

Both treatment-minus-control directions improved the anchor on all five outer folds at small positive weights. Their OOF residual correlation is 0.6865, so the directions are related but not redundant.

## Nested composition

The production composer does not tune weights on the fold it scores. For each held fold it chooses weights on the other four folds from fixed grids that both include zero:

- LightGBM: `0, 0.15, 0.20, 0.25, 0.30`
- XGBoost: `0, 0.05, 0.10, 0.15`

Held-fold selections were:

| Held fold | LGB weight | XGB weight |
| ---: | ---: | ---: |
| 0 | 0.25 | 0.10 |
| 1 | 0.20 | 0.10 |
| 2 | 0.20 | 0.10 |
| 3 | 0.20 | 0.10 |
| 4 | 0.20 | 0.10 |

The cross-fitted honest OOF AUC is **0.9681718649**, a **+0.0002865868** gain over the quad anchor. Every held fold improves. Coordinate-wise median deployment weights are **0.20 LGB + 0.10 XGB**, giving a full OOF AUC of **0.9681744753**.

A paired DeLong test on the cross-fitted honest prediction gives standard error approximately **1.62e-5**, a 95% interval for the AUC gain of approximately **[+0.0002548, +0.0003184]**, and p approximately **8.8e-70**. This is a paired comparison of the same 691k rows, not an independent-model confidence interval.

## Stability gates

The frozen deployment correction is positive in every tested slice:

- modulo-ID partitions: 2/2, 3/3, 5/5, 7/7, and 11/11 wins;
- contiguous sorted-ID partitions: 5/5, 10/10, and 20/20 wins;
- the worst measured contiguous 20-block delta is still positive at approximately **+0.0001514 AUC**.

The composer therefore fails closed: if nested held-fold performance, the frozen deployment OOF gain, or any configured stability family stops being uniformly positive, deployment residual weights are set to zero and the anchor is emitted instead.

## Current submission artifact

The locally reconstructed candidate contains **296,302** rows and has SHA-256:

`562dc0bc7ed6a977942f421560b5149d83419d171494a57618a5cdff69f1f07d`

This is the strongest solution supported by the current offline evidence. It is not described as a guaranteed competition winner because the competition leaderboard is deliberately not used for selection in this campaign.
