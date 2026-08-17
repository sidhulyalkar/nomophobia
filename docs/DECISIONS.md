# S6E8 Frontier v3 — Append-only decision log

Promotion contract: S0 is plumbing/direction only; S1 may advance but never promote; S2 estimates effect size; only S3 = full 691,369 rows × 3 seeds × 5 folds can PROMOTE. A promotion requires ≥13/15 positive fold-seed deltas, pooled paired-bootstrap CI above zero, positive test-missingness-weighted delta, and candidate fold-AUC std ≤120% of baseline.

| ID | hypothesis | tier | Δ AUC | CI95 | folds+ | verdict | note |
|---|---|---:|---:|---|---:|---|---|
| V2-FE | pre-audit engineered +0.00449 | historical | +0.00449 | n/a | n/a | **INVALIDATED** | A1/A4/A5 confounded; do not cite as v3 evidence. |
| V2-META | pre-audit nested meta +0.0018 | historical | +0.0018 | n/a | n/a | **INVALIDATED** | A2/A3/A4 confounded; equal-data control required. |
| A1-FE-120K | native categorical + scale-safe engineered representation vs raw | S1 | **+0.004157** | [+0.003442,+0.004871] | 5/5 | **ADVANCE → S2** | 0.951392 → 0.955549. Strong audited representation result. |
| A2-META-20K | rank-only meta beats equal-data control / optimized blend | S0 | -0.000185 | [-0.001617,+0.001138] | 2/5 | **KILL DEFAULT META** | Old meta promotion does not survive equal-data control. |
| G2 | hidden coarse value lattice exists | full forensics | n/a | n/a | n/a | **KILL** | Values are essentially a clean 0.01 grid. |
| G1 | joint digit tuple reveals shared latent generator component | S0 | -0.008557 | [-0.01238,-0.00483] | 0/5 | **KILL** | Max pair MI <0.01 nats; joint digit categories hurt. |
| G5-COVERAGE | partial row-vector keys recur | full forensics | n/a | n/a | n/a | **PASS GATE** | Best triple has 15.19% test coverage at train count ≥5. |
| G5-EB | fold-safe empirical-Bayes group target means exploit recurring keys | S0 | -0.009024 | [-0.009949,-0.008105] | 0/5 | **KILL** | Coverage is real; target-mean lookup is the wrong exploitation mechanism. |
| H1 | cross-fitted missing-pattern isotonic fixes global cross-regime ranking | S0 | -0.001162 | [-0.001404,-0.000959] | 0/3 | **KILL** | Regime calibration moves consistently backward. |
| H3-120K | test-pattern masking augmentation improves raw backbone | S1 | +0.000600 | [+0.000470,+0.000771] | 5/5 | PARK | Positive against underfit backbone; required stronger control. |
| H3-350K-STRONG | masking survives a stronger backbone | S2 | **-0.000331** | [-0.000400,-0.000264] | 0/5 | **KILL** | Earlier gain was regularization compensation. |
| A6-RECON | corrected symmetric generator reconstruction helps | S0 | +0.000076 | [-0.000689,+0.000873] | 3/5 | **PARK** | Fix removes old negative result but effect is neutral. |
| G4-GATE | nearest-original 4-level severity carries label information | S0 forensic | n/a | n/a | n/a | **PASS GATE** | Standalone parent-severity AUC ≈0.874; MI ≈0.21 nats. |
| G4-FE | append parent severity/distance/entropy | S0 | -0.001286 | [-0.002175,-0.000660] | 1/5 | **KILL** | Strong source prior is globally redundant. |
| G4-BLEND | blend nearest-original severity as independent expert | S0 | ~0 | [-0.000016,+0.000014] | 2/5 | **KILL** | Source signal does not improve competition backbone globally. |
| H2-60K | daily-missing multiple-imputation marginalization | S0 | +0.006606 | [+0.001410,+0.012205] | 4/5 | ADVANCE | Large exploratory result demanded S1 replication. |
| H2-120K | H2 replicates at S1 | S1 | **-0.005807** | [-0.009250,-0.002208] | 0/5 | **KILL** | S0 win reversed completely. |
| O1 | pairwise lambdarank supplies useful decorrelated expert | S0 | -0.048515 | [-0.05244,-0.04540] | 0/5 | **KILL** | Far too weak despite decorrelation. |
| A1-FE-v31-120K | optimized full-reference engineered representation vs raw | S1 | **+0.005364** | [+0.004736,+0.006122] | 5/5 | **ADVANCE → S2** | 0.948702 → 0.954066. |
| G3-KMEANS8 | transductive latent components add incremental signal | S0 | -0.000392 | [-0.001537,+0.000645] | 0/5 | **PARK** | Components largely recover exposure structure. |
| O2-INIT | source ordinal severity as LightGBM init_score | S0 | -0.001999 | [-0.005142,+0.000768] | 0/5 | **PARK** | Direction negative. |
| O3-ORDINAL | four-level source pseudo-severity auxiliary expert | S0 | +0.000141 blend | [-0.000184,+0.000577] | 3/5 | **PARK** | Weak standalone; unresolved tiny blend value. |
| O4-FOCAL-10K | focal loss creates useful error diversity | S0 | +0.000919 | [-0.000213,+0.002165] | 4/5 | ADVANCE → S1 | Attractive tiny-screen result. |
| O4-FOCAL-120K | focal loss replicates | S1 | **-0.000214** | [-0.000462,+0.000007] | 2/5 | **KILL** | S0 gain vanishes. |
| E1-RAW+COMBINED | raw model adds diversity to corrected combined model | S1 | **+0.000481** | [+0.000241,+0.000721] | 4/5 | **ADVANCE → S2** | Early dual-view evidence. |
| V32-ITER-340 | 340 trees beats 260 on identical eval rows | S1 | **+0.001134** | [+0.000908,+0.001344] | 5/5 eval subpartitions | **ADVANCE** | Low-tree screens underfit the winning representation. |
| V32-HC700-S16 | 700-tree combined/raw rank blend | S1 | **+0.000890** | [+0.000497,+0.001296] | 5/5 eval subpartitions | **ADVANCE** | 67.5/32.5 combined/raw. |
| V32-HC700-S17 | 700-tree dual-view replication | S1 | **+0.000658** | [+0.000305,+0.001069] | 5/5 eval subpartitions | **ADVANCE** | 62.5/37.5. |
| V32-HC700-S18 | 700-tree dual-view replication | S1 | **+0.000824** | [+0.000390,+0.001237] | 5/5 eval subpartitions | **ADVANCE** | 62.5/37.5. |
| V32-HC1000-S16 | 1000-tree combined/raw rank blend | S1 | **+0.001043** | [+0.000601,+0.001511] | 5/5 eval subpartitions | **ADVANCE → S2** | 62.5/37.5; strongest complete local run. |
| V32-HC1000-S17 | 1000-tree dual-view replication | S1 | **+0.001059** | [+0.000589,+0.001519] | 5/5 eval subpartitions | **ADVANCE → S2** | 57.5/42.5. |
| V32-HC1000-S18 | third 1000-tree replication | S1 | n/a | n/a | n/a | **NO RESULT** | Runtime ended before raw expert finished. |
| V33-AUDIT-220 | v3.2 diversity kills valid at mature capacity | audit | n/a | n/a | n/a | **VOID / RETRIAL REQUIRED** | Evidence/XGB/router/capacity verdicts were based on the same 220-tree underfit run. |
| V33-AUDIT-ZEROW | selector weight=0 proves no diversity value | audit | n/a | n/a | n/a | **REJECT DECISION RULE** | `[0,0]` from comparing baseline to itself is not evidence. Use forced 5/10/15% tests. |
| V33-FRAMING | digit artifacts are load-bearing | audit | +0.000200 all-digit block | n/a | n/a | **REJECT FRAMING** | Frequency ablation costs ~0.003832 and behavior ~0.001759. |
| V33-FREQ-SMOKE | train+test reference adds signal without source asymmetry | S0 | +0.000356 | [-0.000810,+0.001526] | 3/5 eval subpartitions | **PLUMBING / NO CLAIM** | 12k/180-tree smoke; source adversarial AUC=0.5378. |
| V33-REPRO | graveyard experiments reproducible | engineering | n/a | n/a | n/a | **FIXED** | Restored generating scripts and forced-weight tools. |
| NOM-FREQ-SOURCE-100K | frequency block encodes dangerous source membership beyond known missingness shift | full source forensic | overall source AUC 0.56529 | n/a | n/a | **PASS SAFETY GATE** | Missingness-only source AUC=0.56627 and complete-row frequency source AUC=0.49924. Detectable source shift is essentially explained by missingness. |
| NOM-FREQ-TX-60K | train+test frequency reference improves target ranking over train-only reference | S0 directional | **+0.000550** | [+0.000036,+0.001113] | 4/5 eval subpartitions | **ADVANCE → S1** | 700-tree combined63, disjoint 45k train/15k eval. Small transductive benefit; below S1 scale. |
| NOM-SUB-V01 | mature dual-view full-data model produces valid first competition submission | submission engineering | n/a | n/a | n/a | **READY** | All 691,369 labels; 296,302/296,302 test rows, 0 NaNs. 1000 trees each; 62.5/37.5 rank blend. |
