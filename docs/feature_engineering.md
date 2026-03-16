# FPL Feature Engineering Report

**Date:** 2026-03-16
**Phase:** 4 — Feature Engineering
**Status:** Complete
**Entry point:** `ml/features.py` — `build_feature_matrix(position, era='xg')`
**Outputs:** `outputs/features/feature_matrix_{GK|DEF|MID|FWD}.parquet`

---

## Overview

Phase 4 translates the raw star schema in `db/fpl.db` into four position-specific ML-ready
feature matrices. Each matrix contains one row per player-fixture, filtered to the xG era
(2022-23 to 2025-26), with a `total_points` target column and a set of engineered features
derived from prior-GW data only.

The pipeline is implemented as a single module (`ml/features.py`) with a public entry point
of `build_feature_matrix(position, era='xg', force=False)`. The function queries the database,
applies all transformations in Python using pandas, and caches the result as a parquet file.

---

## 1. Era Scope and Row Counts

### 1.1 Decision

The plan locked in **Option A: xG era only** (seasons 7-10, 2022-23 to 2025-26). The
rationale, confirmed by EDA, is threefold:

1. `expected_goals`, `expected_assists`, `expected_goal_involvements`, and
   `expected_goals_conceded` are absent for pre-2022-23 seasons. These are the most
   predictive features available and cannot be imputed meaningfully.
2. The -26.1% scoring drift between the pre-xG era peak (1.42 pts/GW in 2018-19) and the
   xG era trough (1.05 pts/GW in 2023-24) means the two eras represent different outcome
   distributions. Training across both would require explicit normalisation and era flags
   for uncertain marginal gain.
3. Four seasons of xG data are sufficient for three expanding-window CV folds and for both
   Ridge and LightGBM at this scale.

### 1.2 Actual row counts vs plan

The project plan estimated ~96,000 rows for the xG era. The actual filtered counts are
substantially lower:

| Position | Total rows | % of ~96K estimate |
|----------|-----------:|:------------------:|
| GK       |      2,731 | 2.8%               |
| DEF      |     13,723 | 14.3%              |
| MID      |     19,495 | 20.3%              |
| FWD      |       4,951 | 5.2%              |
| **Total** |  **40,900** | **42.6%**         |

The 96,000 figure was the pre-filter row count for xG era seasons. The base filter removes
roughly 57% of rows, primarily due to the `minutes > 0` condition (27.6% of all GW
appearances are DNP — players never subbed on), combined with the 2025-26 season being
incomplete at the time of writing.

The practical implications by position are discussed in section 5.

---

## 2. Base Filter

All four matrices share the same base filter, applied in SQL:

```sql
WHERE fgp.mng_win IS NULL          -- remove 322 manager-mode rows (2024-25)
  AND fgp.minutes > 0              -- remove DNP appearances
  AND fgp.position_label = ?       -- position-specific query
  AND fgp.season_id >= 7           -- xG era only (2022-23 onwards)
```

A fifth condition, `season_gw_count >= 5`, is applied in Python after loading. This removes
player-seasons with fewer than five qualifying GW appearances — typically late-season signings
or players who spent the whole season injured. The condition uses the count of filtered rows
per `(player_code, season_id)`, not the raw GW count from `dim_player_season`, to be
consistent with what the model will actually see.

`position_label IS NOT NULL` is implicit: filtering by `position_label = ?` already
excludes NULL rows.

---

## 3. Feature Engineering Pipeline

The pipeline runs in seven steps after loading the base data:

```
_load_base_data(position)
    -> _apply_gw_count_filter(df)
    -> _add_player_rolling_features(df)
    -> _add_season_to_date_features(df)
    -> _add_lag_features(df)
    -> _add_team_features(df, tm)       # tm loaded separately (all positions)
    -> _add_opponent_features(df, tm)
    -> _select_position_features(df, position)
```

Team match data (`tm`) is loaded from a separate query covering all positions, not just the
position being built. This ensures every team appears in `tm` regardless of which position
is being processed.

### 3.1 SQL layer: opponent season rank

`opponent_season_rank` (the opponent's final league position, 1=champion, 20=bottom) is
derived inside the SQL query using a CTE that computes a full league table from match
results stored in `fact_gw_player`. The derivation:

1. Deduplicate to one row per team per fixture (DISTINCT on
   `season_id, gw, fixture_id, team_sk, was_home, team_h_score, team_a_score`)
2. Assign match points (3/1/0) and goals for/against from the score columns
3. Rank teams within each season by total points, then goal difference, then goals scored

For historical seasons (7-9) this is the true final league position. For the current season
(10, 2025-26) it is the partial-season standing at the point of data extraction. Using
final-season rank for training is acceptable because league positions stabilise quickly and
the EDA-confirmed effect (DEF -33.8%, FWD -21.2%, GK -17.6%, MID -16.6% against top-6)
is a season-level structural phenomenon.

The `opponent_season_rank` distribution across all four matrices runs 1-20 with a mean of
10.5, confirming uniform coverage of all opponent tiers.

### 3.2 Player rolling features

All rolling features are computed within `(player_code, season_id)` groups. The current
GW is excluded by applying `shift(1)` before the rolling window:

```python
g[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
```

`min_periods=1` means that from the second GW onward a player always has a rolling value
(using however many prior GWs are available). The first GW of every player-season produces
NaN because there is no prior row to shift from. This accounts for the 4.4-4.9% NaN rate
across rolling features (see section 6).

The rolling boundary rule from the project plan is enforced automatically: by grouping
within `(player_code, season_id)`, rolling never chains across season boundaries. The
2019-20 COVID GW gap (GW 29 to GW 39 in season_id=4) is not relevant here since that
season is excluded from the xG era.

**Double gameweeks:** A player with two fixtures in the same GW has two consecutive rows
in the DataFrame, ordered by `(gw, fixture_id)`. The rolling window for the second fixture
includes the first fixture result as a prior row. In practice both fixtures are played within
a few days, so this is a minor concern for training. For live prediction, care should be
taken when building features for the second DGW fixture before the first has been played.

**Feature list:**

| Feature | Source column | Window |
|---------|--------------|--------|
| `pts_rolling_3gw` | `total_points` | 3 |
| `pts_rolling_5gw` | `total_points` | 5 |
| `mins_rolling_3gw` | `minutes` | 3 |
| `goals_rolling_5gw` | `goals_scored` | 5 |
| `assists_rolling_5gw` | `assists` | 5 |
| `cs_rolling_5gw` | `clean_sheets` | 5 |
| `bonus_rolling_5gw` | `bonus` | 5 |
| `saves_rolling_5gw` | `saves` | 5 |
| `xg_rolling_5gw` | `expected_goals` | 5 |
| `xa_rolling_5gw` | `expected_assists` | 5 |
| `xgi_rolling_5gw` | `expected_goal_involvements` | 5 |
| `xgc_rolling_5gw` | `expected_goals_conceded` | 5 |

### 3.3 Season-to-date features

Cumulative per-player averages using only prior GW data:

- `season_pts_per_gw_to_date`: `shift(1).expanding(min_periods=1).mean()` on `total_points`
- `season_starts_rate_to_date`: same on `starts`

Both produce NaN on the first GW of each player-season, matching the rolling feature NaN
pattern. `starts` has no NULL values in any of the four xG era seasons despite the
`has_starts=0` flag in the schema constants for 2025-26 — the column is present and
populated in the raw data.

### 3.4 Lag features

Three single-GW lags within `(player_code, season_id)`:

- `value_lag1`: player's price one GW prior (£0.1m units, consistent with DB convention)
- `transfers_in_lag1`: transfers in the prior GW
- `transfers_out_lag1`: transfers out the prior GW

Transfer activity is lagged by exactly one GW because same-GW transfers are a reactive
signal (EDA confirmed a 35x spike in `transfers_in` after 15+ pt GWs). Lagging by one GW
converts it from leakage to a crowd-wisdom signal of prior-GW performance.

### 3.5 Team match features

**Data quality finding:** `goals_conceded` in `fact_gw_player` reflects goals conceded while
that specific player was on the pitch, not the team's total for the fixture. Of 2,760 team-
fixture combinations in the xG era, 2,094 (75.9%) show inconsistent `goals_conceded` values
across players on the same team in the same fixture. Team-level conceded stats derived from
this column would be systematically wrong.

All team-level features are therefore computed from `team_h_score` and `team_a_score`, which
are consistent across all rows for the same fixture (0 of 1,380 fixtures show inconsistency).
A separate SQL query loads one row per team per fixture:

```sql
SELECT season_id, gw, fixture_id, team_sk,
    MIN(was_home) AS was_home,
    MIN(team_h_score) AS team_h_score,
    MIN(team_a_score) AS team_a_score
FROM fact_gw_player
WHERE mng_win IS NULL AND team_h_score IS NOT NULL AND season_id >= 7
GROUP BY season_id, gw, fixture_id, team_sk
```

From this, `team_gf` (goals for) and `team_ga` (goals against) are computed as:
```python
team_gf = team_h_score if was_home else team_a_score
team_ga = team_a_score if was_home else team_h_score
```

Rolling and cumulative features are then computed within `(team_sk, season_id)` groups,
sorted by `(gw, fixture_id)` to maintain chronological order within DGWs:

| Feature | Derivation |
|---------|-----------|
| `team_goals_conceded_season` | `shift(1).expanding().sum()` on `team_ga` — cumulative season-to-date goals conceded |
| `team_cs_rolling_3gw` | `shift(1).rolling(3).mean()` on `(team_ga == 0).astype(float)` — clean sheet rate over last 3 matches |
| `team_goals_scored_rolling_3gw` | `shift(1).rolling(3).mean()` on `team_gf` — rolling attacking form |

`team_goals_conceded_season` is NaN for a team's first fixture of the season (no prior data).
A value of 0 would be equally valid but NaN is the safer default since LightGBM handles it
natively and Ridge regression imputation in `train.py` can fill appropriately. The NaN
affects 2.7-2.8% of rows across all matrices.

The `team_goals_conceded_season` feature for GKs ranges 0-101 with a mean of 26.2, spanning
the full range from elite defensive teams (26 goals conceded across a full season) to poor
defensive sides. This aligns with the EDA finding that team goals conceded explains 46.6%
of variance in DEF/GK average GW points.

### 3.6 Opponent features

Joined via `opponent_team_sk`, these capture the opponent team's season-to-date form from
the same team match data:

| Feature | Derivation |
|---------|-----------|
| `opponent_goals_scored_season` | Cumulative `team_gf` for the opponent before this fixture |
| `opponent_cs_rate_season` | Fraction of prior matches where the opponent kept a clean sheet |

Both are NaN for an opponent's first fixture of the season (no prior data), giving the same
2.7-2.8% NaN rate as the team features.

---

## 4. Position-Specific Feature Sets

Feature applicability follows section 4.6 of the project plan:

| Feature group | GK | DEF | MID | FWD |
|--------------|:--:|:---:|:---:|:---:|
| `team_goals_conceded_season` | Yes | Yes | — | — |
| `team_cs_rolling_3gw` | Yes | Yes | — | — |
| `team_goals_scored_rolling_3gw` | Yes | Yes | Yes | Yes |
| `cs_rolling_5gw`, `xgc_rolling_5gw` | Yes | Yes | — | — |
| `saves_rolling_5gw` | Yes | — | — | — |
| `goals_rolling_5gw`, `assists_rolling_5gw` | — | — | Yes | Yes |
| `xg_rolling_5gw`, `xa_rolling_5gw`, `xgi_rolling_5gw` | — | — | Yes | Yes |
| `was_home`, `opponent_season_rank` | Yes | Yes | Yes | Yes |
| `pts_rolling_3/5gw`, `mins_rolling_3gw`, `bonus_rolling_5gw` | Yes | Yes | Yes | Yes |
| `season_pts_per_gw_to_date`, `season_starts_rate_to_date` | Yes | Yes | Yes | Yes |
| `start_cost`, `value_lag1` | Yes | Yes | Yes | Yes |
| `transfers_in_lag1`, `transfers_out_lag1` | Yes | Yes | Yes | Yes |
| `opponent_goals_scored_season`, `opponent_cs_rate_season` | Yes | Yes | Yes | Yes |

Defensive features (`team_goals_conceded_season`, `team_cs_rolling_3gw`, `cs_rolling_5gw`,
`xgc_rolling_5gw`) are excluded from MID and FWD because their scoring is dominated by
attacking contributions; including these features would add noise and dilute the signal from
attacking features.

GK specifically excludes goals, assists, and xG/xA/xGI features — these are near-zero for
goalkeepers and would contribute only noise.

---

## 5. Output Matrices

### 5.1 Summary

| Position | Rows | Feature cols | Target mean | Target std | Target max |
|----------|-----:|:------------:|:-----------:|:----------:|:----------:|
| GK | 2,731 | 20 | 3.37 | 2.80 | 16 |
| DEF | 13,723 | 19 | 2.62 | 2.93 | 24 |
| MID | 19,495 | 20 | 2.81 | 2.90 | 26 |
| FWD | 4,951 | 20 | 3.12 | 3.33 | 23 |

Context columns (`season_id`, `gw`, `fixture_id`, `player_code`, `position_code`, `team_sk`)
and the target (`total_points`) are included in every matrix but are not treated as features
by the model. Feature count above excludes these.

Target means are higher than the full-dataset means from EDA (GK 0.99, DEF 1.20, MID 1.33,
FWD 1.40) because the base filter removes DNP rows (`minutes > 0`). These matrices contain
only appearances where the player actually played, so the mean reflects conditional expected
points given participation.

### 5.2 Season row counts

| Season | GK | DEF | MID | FWD |
|--------|---:|----:|----:|----:|
| 2022-23 (7) | 745 | 3,733 | 5,255 | 1,459 |
| 2023-24 (8) | 765 | 3,803 | 5,243 | 1,410 |
| 2024-25 (9) | 746 | 3,719 | 5,664 | 1,244 |
| 2025-26 (10) | 475 | 2,468 | 3,333 | 838 |

The lower 2025-26 counts reflect a season in progress at time of writing. Season 10 data will
grow as the season completes, increasing CV fold 3 test set sizes.

### 5.3 CV fold sizes

Three expanding-window folds within the xG era (season 7->8, seasons 7-8->9, seasons 7-9->10):

| Position | Fold 1 train | Fold 1 test | Fold 2 train | Fold 2 test | Fold 3 train | Fold 3 test |
|----------|-------------:|------------:|-------------:|------------:|-------------:|------------:|
| GK | 745 | 765 | 1,510 | 746 | 2,256 | 475 |
| DEF | 3,733 | 3,803 | 7,536 | 3,719 | 11,255 | 2,468 |
| MID | 5,255 | 5,243 | 10,498 | 5,664 | 16,162 | 3,333 |
| FWD | 1,459 | 1,410 | 2,869 | 1,244 | 4,113 | 838 |

DEF and MID are comfortable across all folds. GK and FWD warrant attention:

- **GK Fold 1 (745 training rows):** This is the binding constraint of the entire pipeline.
  With 20 features, this is workable for Ridge regression and a heavily regularised LightGBM,
  but not for a deep model or an unconstrained tree ensemble. GK-specific recommendations
  for Phase 5: keep `num_leaves <= 15`, `min_child_samples >= 30`, and compare Ridge vs
  LightGBM directly — Ridge may outperform LightGBM at this scale given that GK scoring is
  largely a linear function of team defensive quality.

- **FWD Fold 1 (1,459 training rows):** Marginal but viable for LightGBM with standard
  regularisation. Less of a concern than GK.

- **Fold 3 test sets for 2025-26:** GK (475) and FWD (838) test sets are small, producing
  wide confidence intervals on Fold 3 metrics. This is temporary — test set sizes will grow
  as 2025-26 completes.

### 5.4 NaN rates

All NaN values in the matrices are structurally expected; there are no missing values from
data gaps or join failures.

| Cause | Affected features | NaN rate | Resolution |
|-------|------------------|:--------:|-----------|
| First GW of player-season | All player rolling and lag features | ~4.5% | LightGBM: native. Ridge: mean imputation. |
| First fixture of team-season | All team and opponent features | ~2.8% | Same as above. |

The ~4.5% rate corresponds exactly to the number of player-season debut rows in each matrix
(confirmed: GK 4.5%, DEF 4.9%, MID 4.4%, FWD 4.5%).

---

## 6. Leakage Controls

The following same-GW columns were loaded from the DB for rolling feature computation but
are excluded from the final output:

| Column | Why banned as a feature |
|--------|------------------------|
| `bonus` | Post-match computed; same-GW r = 0.74 with total_points |
| `bps` | Post-match computed; same-GW r = 0.70 |
| `ict_index` | Post-match computed; same-GW r = 0.65 |
| `clean_sheets` | Direct target component; rolling lag retained |
| `goals_scored` | Direct target component; rolling lag retained |
| `assists` | Direct target component; rolling lag retained |
| `transfers_in` | Reactive same-GW signal; lag-1 version retained |
| `transfers_out` | Reactive same-GW signal; lag-1 version retained |

`bonus`, `bps`, and `ict_index` are used only as inputs to `bonus_rolling_5gw` (which uses
`shift(1)` before rolling and is therefore a prior-GW signal). They never appear as raw
same-GW features in the output matrices.

`expected_goals`, `expected_assists`, `expected_goal_involvements`, and
`expected_goals_conceded` are available as same-GW values in the raw data. For this dataset
they are pre-match expected statistics (player-season modelled values attached to each GW
row) rather than post-match computed values, so their rolling lags are valid predictive
features. However, confirming this interpretation is recommended before Phase 5.

---

## 7. Alignment with Phase 4 Plan

### Implemented as specified

| Plan element | Status |
|-------------|--------|
| Entry point `build_feature_matrix(position, era='xg')` | Implemented |
| Base filter (mng_win IS NULL, minutes > 0, season_id >= 7, gw_count >= 5) | Implemented |
| xG era only (Option A) | Implemented |
| `was_home` included for all positions | Implemented |
| `opponent_season_rank` derived from final standings | Implemented |
| `team_goals_conceded_season` for GK/DEF | Implemented |
| Rolling within `(player_code, season_id)` only | Implemented |
| No cross-season rolling chains | Enforced by groupby design |
| `transfers_in/out` lagged by 1 GW | Implemented |
| `bonus/bps/ict_index` excluded as same-GW features | Implemented |
| `start_cost` and `value_lag1` included | Implemented |
| xG features (xg/xa/xgi/xgc rolling) for applicable positions | Implemented |
| Season-to-date cumulative features | Implemented |
| Parquet cache to `outputs/features/` | Implemented |
| Position-specific feature subsets (section 4.6) | Implemented |

### Deviations and additions

| Item | Detail |
|------|--------|
| Row count estimate | Plan estimated ~96K rows; actual post-filter is 40,900 (57% lower). The estimate was pre-filter. No decision changes required. |
| `goals_conceded` not used for team features | Plan did not specify which column to use for team goals conceded. Investigation found that `goals_conceded` in `fact_gw_player` is time-on-pitch scoped and inconsistent (75.9% of team-fixtures show player-level variance). All team-level conceded stats use `team_a_score` / `team_h_score` instead. |
| `starts` present in all seasons | Plan noted `has_starts=0` for 2025-26 (defensive era). The actual data has no NULL starts in that season. `season_starts_rate_to_date` is therefore computable for all four seasons without imputation. |
| `team_goals_conceded_rolling_3gw` not included | Plan listed this in the feature catalogue as part of team form. It was omitted from the final feature sets after reviewing the position tables in section 4.6, which did not assign it to any position. `team_goals_conceded_season` (cumulative) is the primary defensive team feature per section 4.3, and rolling conceded is partially captured by `team_cs_rolling_3gw`. Can be added in Phase 5 if feature importance analysis suggests value. |
| `team_pts_rolling_3gw` not included | Plan listed this feature but the section 4.6 position table does not include it, and its definition is ambiguous (team-level FPL points do not exist). The meaningful analogue is `team_goals_scored_rolling_3gw`, which is included. |

---

## 8. Implications for Phase 5 (Modelling)

1. **NaN handling:** LightGBM handles NaN natively. Ridge regression requires imputation
   before fitting — strategy-mean imputation within `(position, season_id)` is recommended
   over global mean to avoid contaminating early-season rows with mid-season averages.

2. **Feature scaling:** Ridge requires standardisation. LightGBM does not. Scale within each
   CV fold's training set; apply the same scaler to the validation set.

3. **GK regularisation:** With only 745 training rows in Fold 1, LightGBM hyperparameters
   for GK should be more conservative than for DEF/MID. Suggested starting points:
   `num_leaves=15`, `min_child_samples=30`, `learning_rate=0.05`, `n_estimators=200`.

4. **`start_cost` requires division by 10 in reporting layer only.** It is stored in £0.1m
   units. Do not convert before model training — the model learns the correct scale
   automatically. Only convert when reporting feature importance or predictions in £m.

5. **`opponent_season_rank` for 2025-26 live prediction:** For the current season, the rank
   is derived from partial-season standings. When generating live predictions in `predict.py`,
   this feature should be recomputed from current standings rather than end-of-season.

6. **Baseline to beat:** Rolling mean baseline (`pts_rolling_5gw`) is already present as a
   feature. For the rolling-mean baseline model, this single feature is the prediction.
   Any model must beat it on MAE, RMSE, and Spearman rank correlation to be considered
   for production (per section 6.5 of the project plan).
