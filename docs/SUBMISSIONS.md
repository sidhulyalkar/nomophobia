# Submission log

This file tracks leaderboard-facing artifacts separately from OOF research evidence.

| Version | Model | Status | Public LB | Notes |
|---|---|---|---:|---|
| v0.1 | 62.5% combined rank + 37.5% raw rank, 1000 trees each | **ready** | pending | Initial full-data S1-derived submission |
| v0.1-control | combined-only, 1000 trees | **ready** | pending | Measures whether the raw-view hedge helps on LB |
| v0.1-raw325 | 67.5% combined rank + 32.5% raw rank | reserve | pending | Nearby weight probe only if first two submissions are informative |
| v0.1-raw425 | 57.5% combined rank + 42.5% raw rank | reserve | pending | Nearby weight probe only if first two submissions are informative |

Public scores are observational. Promotion decisions remain OOF-driven.

## Local full-data generation record

Initial v0.1 was generated against **all 691,369 training rows** and all **296,302 test rows** using 1000 trees per expert.

- combined fit + inference: ~103.3 s
- raw fit + inference: ~26.0 s
- test probability correlation between experts: ~0.98467
- test rank correlation between experts: ~0.99002
- output rows validated against `sample_submission.csv`: 296,302 / 296,302
- missing/non-finite predictions: 0
- primary CSV SHA-256: `1ca9d554cbab65a3e0d6a919b4cbd52b497306710cdc59b074f833874d91755f`

The full-data test-rank correlation is somewhat higher than in the 120k OOF screens, so the leaderboard control is useful as a weak observation about whether the raw hedge survives at full-data fit scale. The S3 campaign remains authoritative.

## First daily submission plan

Use the 10-submission allowance as an experiment budget:

1. **Primary:** v0.1 62.5/37.5 dual-view rank blend.
2. **Control:** combined-only 1000-tree model.
3. Submit at most one neighboring weight only if the first two scores create a useful hypothesis.
4. Reserve the remaining slots for production-scale S3 / mature diversity / public-stack candidates.

Suggested Kaggle descriptions:

- `NOMOPHOBIA v0.1 | full 691k | combined1000 + raw1000 | rank 62.5/37.5`
- `NOMOPHOBIA v0.1 control | full 691k | frequency+behavior LGBM63 | 1000 trees`
