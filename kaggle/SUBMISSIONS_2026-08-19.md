# Five-submission leaderboard campaign — 2026-08-19

This folder is generated from two measured frontier endpoints:

- **Anchor:** `agent/frontier-097-campaign:kaggle/submission_nomophobia_v19_lgb015.csv`, the current selected v19 public-anchor + LGB residual submission.
- **Contrast endpoint:** GitHub Actions artifact `9346320480`, `submission_candidate.csv`, from the LGB identity-screen frontier arm. Its residual gate passed all five held folds at the selected residual weight.

The builder produces five **distinct** 296,302-row Kaggle CSVs and validates schema, ID order, finiteness, non-degeneracy, hashes, and pairwise rank correlation before committing them.

## Recommended submission order

1. `01_v19_lgb015_anchor.csv` — current campaign selection; establishes the anchor score.
2. `02_anchor95_lgb05.csv` — conservative 5% identity-screen probe.
3. `03_anchor80_lgb20.csv` — moderate 20% probe; tests whether the auxiliary family transfers more strongly to the hidden leaderboard.
4. `04_anchor50_lgb50.csv` — balanced diversity probe; maps the middle of the response curve.
5. `05_lgb_identity_screen.csv` — independent LGB endpoint; maximizes diagnostic value and tells us whether the family itself transfers.

The blends are made in rank space because the competition objective is ranking-sensitive. The purpose is not to pretend five tiny variations are five independent discoveries; it is to spend five leaderboard slots on an interpretable response curve between the strongest current anchor and a genuinely different measured frontier endpoint.

After scores arrive, record them next to these five files. The shape of the response curve tells us whether to stay near the v19 anchor, increase LGB-family ownership, or kill this direction and spend the next campaign elsewhere.

Generation is handled by `scripts/build_leaderboard_five.py` and `.github/workflows/build_five_kaggle_submissions.yml`.
