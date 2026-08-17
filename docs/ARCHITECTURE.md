# Architecture

## Current winner candidate

The current backbone is deliberately small at the model-family level and rich at the representation level.

### Expert A: transductive-frequency + behavior LightGBM

`lgb_combined63` sees:

- original predictors;
- behavioral composition and burden features;
- explicit missingness structure;
- a small digit/rounding block;
- full-reference train+test value-frequency features.

The frequency block is computed without labels. It is transductive because the unlabeled competition test distribution participates in the frequency counts.

### Expert B: raw LightGBM

`lgb_raw63` receives only the original predictors.

This is intentional. Giving both experts the same engineered view causes their rankings to collapse toward one another. The raw expert is valuable because it follows a different set of split boundaries.

### Combination

Because ROC AUC depends on ordering, models are combined in rank space rather than probability space.

During S3, blend weights are selected without observing the held fold. The deployed full-data submission should use S3-derived weights once those exist.

## Why not a giant stack by default?

The external public ecosystem already contains many highly correlated tree models. More models are useful only if they contribute different pairwise rankings.

Every diversity candidate therefore has to report:

- standalone OOF AUC;
- rank correlation to the mature blend;
- forced 5%, 10%, and 15% blend deltas;
- paired confidence intervals.

A selector returning 0% weight is not itself evidence of uselessness.
