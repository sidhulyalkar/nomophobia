# Research learnings

## What survived

### Transductive frequency features

This is the strongest measured feature family. Removing it from the engineered view produced the largest ablation loss seen in the representation study.

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

The most valuable asset in this repository is not a particular feature. It is the ability to say **no** to an attractive result when it fails replication.
