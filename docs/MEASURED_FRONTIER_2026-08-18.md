# Measured 0.97 frontier results — 2026-08-18

This note records the completed no-Kaggle research measurements used to choose the next NOMOPHOBIA submission candidate.

## Self-trained matched contrasts

Both matched representation treatments were trained on the full 691,369-row competition training set using fixed schedules and the same five outer folds as their controls.

| lane | control OOF AUC | treatment OOF AUC | treatment gain | control/treatment rank corr | residual gate | honest residual gain |
|---|---:|---:|---:|---:|---|---:|
| LightGBM, 900 rounds, identity + screen | 0.961691535 | 0.964099988 | +0.002408453 | 0.994024 | PASS at 0.0025 | +0.000004563 |
| XGBoost, 1500 rounds, identity + screen | 0.964029255 | 0.966486775 | +0.002457521 | 0.992599 | PASS at 0.0025 | +0.000004374 |

The representation gain is large and replicated on all five folds for both learners. The residual gain against the already-strong 0.967870056 inner-10 TE anchor is intentionally much smaller, but both directions improved all five folds and both even/odd ID slices under the prospective gate.

## External/public OOF audit

The public 74-stream library was independently ID-audited. Simple equal-weight and positive-only stacks plateau below the strongest published blends:

- best equal-top-k: 0.969310074 OOF at k=8;
- cross-fitted rank-logit: 0.969536215 OOF;
- positive fixed rank-logit: 0.969142540 OOF;
- signed fixed diagnostic: 0.969547312 OOF.

This kills the idea that simply ingesting dozens of public predictions is sufficient. The useful public frontier is concentrated in carefully selected blends and residual directions.

## Strong audited public controls

Two stronger audited anchors were retained:

1. public `18_blend`: 0.969856172 OOF; the source notebook reports a 0.97097 public leaderboard score;
2. public v19 blend: 0.970099209 OOF.

The v19 artifact is the strongest aligned OOF anchor in the current audited set.

## NOMOPHOBIA residuals on the mature anchors

### Conservative 18-blend portfolio

A 7.5% NOMOPHOBIA robust-portfolio correction produced:

- OOF AUC: 0.969862120;
- gain vs public 18_blend: +0.000005948;
- 5/5 positive folds under the NOMOPHOBIA anchor folds;
- 5/5 positive folds under the independent public fixed-schedule folds.

The broader sweep peaked slightly higher near 10%, but 7.5% was retained because it stayed positive across every fold in both systems.

### v19 + LightGBM identity/screen direction

The best current deployment candidate is:

`rank(rank(v19_public_anchor) + 0.015 * NOMOPHOBIA_LGB_identity_screen_direction)`

Measured result:

- v19 control OOF AUC: 0.9700992093622354;
- candidate OOF AUC: **0.9701014001897124**;
- gain: **+0.000002190827477**;
- NOMOPHOBIA anchor fold wins: 4/5;
- public fixed-schedule fold wins: 5/5;
- worst fold delta: -0.000001682, inside the predeclared -0.000002 tolerance;
- even-ID slice: +0.000001479;
- odd-ID slice: +0.000002892;
- selected residual coefficient: 0.015;
- gate: PASS.

A larger coefficient around 0.03 had slightly better pooled OOF but violated the existing worst-fold tolerance and was therefore rejected.

## Submission choice

Recommended attempt:

`submission_nomophobia_v19_lgb015.csv`

Contract:

- 296,302 rows;
- columns exactly `id, addicted_label`;
- 296,302 unique IDs;
- 295,801 unique prediction values;
- no missing/non-finite predictions;
- SHA-256: `e2d0866a57590dce5e20ce4f98edb9cf1822e79a6af7e17cc0be22e887f905a2`.

This is the strongest current OOF candidate, but its leaderboard score is not claimed until separately measured. The public 0.97097 value belongs to the audited `18_blend` control, not to this new file.

## Research conclusion

The main discovery is not the tiny final residual coefficient. It is the replicated +0.0024 to +0.00246 gain from the identity/screen representation under two different tree families. This confirms that numeric identity and screen-allocation geometry are real competition structure. Once the anchor is already near 0.970, however, most of that signal is redundant with mature ensemble members, so the deployable correction naturally shrinks to basis-point scale.

Next research should therefore search for genuinely new residual geometry rather than increase the same residual weight or retune the TE backbone.