# NOMOPHOBIA v0.3 Research Roadmap

The current evidence supports marginal population frequency + behavioral decomposition, with a raw LightGBM hedge that repeatedly helped at S1 but became more correlated on the first full-data test inference. The authoritative 691,369-row 3-seed × 5-fold question remains unresolved.

## Search tree

1. Mature frequency safety: train+test vs train-only target gain, missingness-only source AUC, and complete-row frequency source AUC.
2. Marginal family decomposition with paired CIs.
3. Capacity/diversity curve using a pre-fixed 37.5% raw rank hedge.
4. Higher-order target-free density: selected pair/triple states, value×other-missingness-regime states, and unsigned train/test support stability.
5. Direct low-weight augmentation from the public 7,500-row source labels after exact predictor-overlap removal.
6. Repeated production-scale iteration tuning; freeze the median best count.
7. Authoritative S3 with weight-stability, selection-optimism, seed-spread, and production correlation diagnostics.
8. Mature CatBoost/XGBoost/Evidence forced-weight retrials only after the backbone is frozen.
9. Price Frontier against aligned public OOF streams; `frontier_total_weight` decides whether to keep tuning the backbone.

## Key commands

```bash
python experiments/frequency_stress.py --data-dir /kaggle/input/playground-series-s6e8 --rows 120000 --estimators 1000 --device gpu --out artifacts/frequency_stress.json
python experiments/frequency_family_ablation.py --data-dir /kaggle/input/playground-series-s6e8 --rows 120000 --estimators 1000 --device gpu --out artifacts/frequency_family_ablation.json
python experiments/capacity_diversity_curve.py --data-dir /kaggle/input/playground-series-s6e8 --rows 180000 --estimators 400 700 1000 1400 2000 --device gpu --out artifacts/capacity_diversity_curve.json
python experiments/frequency_geometry.py --data-dir /kaggle/input/playground-series-s6e8 --rows 120000 --estimators 1000 --device gpu --out-dir artifacts/frequency_geometry
python experiments/original_row_augmentation.py --data-dir /kaggle/input/playground-series-s6e8 --original-csv /kaggle/input/<SOURCE>/<source.csv> --rows 120000 --estimators 1000 --device gpu --out-dir artifacts/original_row_augmentation
nomophobia winner --data-dir /kaggle/input/playground-series-s6e8 --out-root artifacts/winner --device gpu --tune-rows 628000 --max-estimators 4000 --patience 200 --tune-repeats 3
nomophobia route --artifact-root artifacts --data-dir /kaggle/input/playground-series-s6e8
```

## Promotion discipline

S1 screens may advance, never promote. S2 estimates effect size. S3 requires 3 distinct 5-fold seeds, 13/15 positive fold-seed deltas, pooled paired CI above zero, positive test-like weighted delta, and controlled fold instability.

If high-capacity raw/combined rank correlation exceeds 0.988 and the raw hedge loses paired benefit, stop fine-grained raw-weight probing and find a genuinely new diversity view.
