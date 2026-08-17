# Research learnings

## What survived

### Transductive frequency features

This is the strongest measured feature family. Removing it from the engineered view produced the largest ablation loss seen in the representation study.

A post-v3.3 source-adversarial audit sharpened the interpretation. On balanced 100k train + 100k test samples, production frequency features distinguish source at about **0.5653 AUC**. Missingness indicators alone achieve about **0.5663**, while the same frequency-source experiment restricted to rows with all original predictors observed falls to **0.4992 AUC**. The measurable source asymmetry is therefore explained by the already-known missingness shift rather than a hidden fingerprint among complete numerical rows.

A separate 60k directional experiment with a 700-tree combined model found train+test reference frequencies ahead of train-only reference frequencies by about **+0.00055 AUC** on an untouched 15k evaluation set, paired interval approximately **[+0.00004,+0.00111]**. This is below S1 scale, so it advances the hypothesis rather than promoting it.

### Behavioral decomposition

Features describing how total screen exposure is composed across social, gaming, work/study, weekend use, and sleep burden add real signal beyond the raw columns.

### High-capacity dual-view modeling

The raw and engineered LightGBM experts become more useful together after both are allowed to mature. Early low-tree-count screens understated this effect.

## What changed our minds

### Decimal digits were interesting, but not load-bearing

The early project was organized around surprising decimal-digit target effects. Later family ablations showed the entire digit/rounding block is worth only a small fraction of the total representation gain.

### More capacity changed diversity decisions

A 220-tree model was far from mature. Some historical XGBoost/Evidence/router kills were therefore voided and moved to mature-capacity retrial.

### Missingness is difficult, but bespoke missingness machinery has not won

Hard specialists, isotonic regime calibration, and marginalization-style imputation experiments failed under larger screens. LightGBM's global missing-value handling remains difficult to beat.

## Methodological lesson

The most valuable asset in this repository is not a particular feature. It is the ability to say **no** to an attractive result when it fails replication, and to distinguish a useful transductive signal from a source-membership artifact before trusting it.
