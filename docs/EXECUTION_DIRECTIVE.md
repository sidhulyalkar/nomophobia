# Corrected Execution Directive

This directive supersedes the older “next run” guidance. It records the audit corrections that changed the campaign.

## Finding 1: 220-tree diversity kills are void

The historical XGBoost, Evidence Expert, advantage-router, and 31-leaf capacity decisions shared the same underfit 220-tree baseline. The project later measured a steep capacity curve and now uses ~700–1000 trees in local high-capacity screens. These decisions therefore require mature retrial.

CatBoost was never properly adjudicated in the old verification set and must be run.

## Finding 2: zero blend weight is not evidence

If a selector chooses candidate weight 0, the candidate blend equals the baseline. A `[0,0]` paired interval is then an identity, not evidence that the candidate has no marginal value.

Diversity members are evaluated with forced candidate weights:

```text
w ∈ {0.05, 0.10, 0.15}
```

or on an S2/S3 selection population large enough to resolve the expected contribution.

## Finding 3: independent replication language

The dual-view headline consists of **five independent S1 replications**:

- 3 at 700 trees, all positive, mean Δ ≈ +0.00079;
- 2 at 1000 trees, all positive, mean Δ ≈ +0.00105.

The five partitions inside one 15k evaluation set are `eval_subpartitions`, not independent replications and must not be counted in the units of the 13/15 S3 rule.

## Finding 4: scientific framing

Feature-family ablations:

| Ablation | Approx. AUC loss |
|---|---:|
| Remove all frequency features | 0.003832 |
| Remove behavioral decomposition | 0.001759 |
| Remove all digit/rounding features | 0.000200 |

The project is therefore framed as **transductive frequency structure + behavioral decomposition**, with a raw-view diversity expert.

Frequency is load-bearing and must be stress-tested before S3 interpretation.

## Finding 5: production-scale iteration tuning

`tune_iterations.py` uses an 88/12 inner split. `--tune-rows 628000` therefore supplies ~552,640 tuning-training rows, closely matching one 5-fold S3 training partition (~553,095 rows).

Default campaign:

```bash
python run_winner.py \
  --data-dir /kaggle/input/playground-series-s6e8 \
  --out-root artifacts/winner \
  --device gpu \
  --tune-rows 628000 \
  --max-estimators 4000 \
  --patience 200 \
  --dry-run
```

If `best_iteration` lands within 10% of `max_estimators`, raise the ceiling and retune.

## Finding 6: reproducibility

The previously missing generating scripts for family diversity, Evidence Expert, advantage routing, and binned target evidence are now shipped under `experiments/`.

# Corrected campaign

## Step 1: frequency stress + production-scale tuning

Run the mature frequency stress control and tune raw/combined expert counts at production scale.

## Step 2: authoritative S3

Three seeds × five folds × all 691,369 rows.

Per seed report:

- exact estimator count per expert;
- expert OOF AUC and fold std;
- honest rotating-selection blend AUC;
- selection AUC and selection optimism;
- rotation weights per held fold;
- raw/combined rank correlation;
- test-missingness-weighted OOF AUC.

Stop and report if seed blend AUCs differ by >0.002 or if production raw/combined rank correlation exceeds 0.988.

## Step 3: mature graveyard retrial

After S3 freezes the backbone, independently tune CatBoost and XGBoost, regenerate Evidence OOF on the same folds, and run forced 5/10/15% tests against the mature S3 blend.

## Step 4: price against the public OOF universe

Assert ID alignment. If the external library has a fold column, Frontier must use the same folds before any meta model. If fold provenance is unavailable, blend only.

Jointly correlation-prune public + Frontier predictions, then run bagged Caruana selection.

Interpret `frontier_total_weight`:

- >=15%: meaningful orthogonal signal;
- 5–15%: marginal, prioritize diversity;
- <5%: stop tuning this LightGBM backbone.

# Reporting contract

Every claim includes tier, `delta_auc`, `delta_ci_95`, `folds_positive`, `delong_p`, `n_effective`, and estimator count.

Always distinguish:

- `independent_replications`: new samples/seeds/models;
- `eval_subpartitions`: partitions of one shared evaluation set.

# Escalation conditions

Stop and report if:

- iteration tuning hits the estimator ceiling;
- two S3 seeds disagree by >0.002 blend AUC;
- S3 dual-view rank correlation exceeds 0.988;
- Frontier receives <5% weight against the public universe;
- any single change measures >+0.003 unexpectedly;
- frequency controls show that the apparent gain is dominated by an unsafe train/test source fingerprint.
