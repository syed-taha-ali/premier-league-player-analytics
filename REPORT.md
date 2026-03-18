# FPL Performance Prediction — End-to-End Data Science Portfolio Report

---

## 1. Executive Summary

This project is a production-grade machine learning pipeline that predicts Fantasy Premier League (FPL) player points on a gameweek-by-gameweek basis. Built across nine structured phases, it ingests a decade of player performance data, engineers predictive features from match statistics and rolling form, trains position-specific regression models, and surfaces results through a six-page interactive Streamlit dashboard.

The final production model is a position-specific Ridge regression, selected over LightGBM and 20 other candidates after rigorous temporal cross-validation. It achieves cross-validated MAEs of 2.13 (GK), 2.14 (DEF), 1.84 (MID), and 2.27 (FWD). A live pipeline fetches current-season data from the FPL API, regenerates predictions after each gameweek, and automatically monitors for model drift against calibrated alert thresholds. As of GW 30 (2025-26 season), all four positions remain well within their thresholds with no alerts raised.

---

## 2. Project Objectives

1. Build a reproducible, end-to-end pipeline from raw CSV data to ranked weekly player predictions.
2. Identify which player and fixture attributes best explain FPL point outcomes, grounded in statistical evidence.
3. Train and rigorously evaluate position-specific models using a leakage-free, temporally sound methodology.
4. Deploy a live GW pipeline that keeps predictions current with the running season.
5. Monitor production model performance over time and support automated retraining at season end.
6. Surface all results in an accessible interactive dashboard suitable for non-technical users.

---

## 3. Data Sources & Description

**Primary source:** [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) — a community-maintained historical dataset covering 10 Premier League seasons (2016-17 to 2025-26).

**Supplementary:** FPL official REST API (`fantasy.premierleague.com/api`) for live current-season data.

**Volume and structure:**
- Raw CSVs: ~197 MB uncompressed across 10 seasons
- Post-ETL database: `db/fpl.db` — 247,308 GW-level player rows at 54 MB
- Modelling subset (xG era, post-filter): 40,900 rows — GK 2,731 | DEF 13,723 | MID 19,495 | FWD 4,951

**Key schema eras (column availability changes across seasons):**

| Era | Seasons | Notable additions |
|-----|---------|-------------------|
| Old Opta | 2016-17 to 2018-19 | Passing, dribble, foul stats; no xG |
| Stripped | 2019-20 | Core stats only; COVID gap (GWs 30-38 never played) |
| Modern core | 2020-21, 2021-22 | Adds position, team, xP |
| xG era | 2022-23 to 2023-24 | Adds xG, xA, xGI, xGC, starts |
| Manager era | 2024-25 | Adds mng_* manager-mode columns |
| Defensive era | 2025-26 | Adds defensive_contribution, clearances, tackles; drops mng_* |

---

## 4. Data Engineering Pipeline

### 4.1 Architecture

The ETL pipeline runs as a Python module (`python -m etl.run`), executing a full drop-and-rebuild of the SQLite database from raw CSVs in approximately 15 seconds. It follows a strict load order to respect foreign key dependencies.

**Star schema — 6 tables:**

| Table | Grain | Primary Key |
|-------|-------|-------------|
| `dim_season` | one row per season | `season_id` |
| `dim_player` | one row per player (cross-season) | `player_code` |
| `dim_team` | team x season | `(season_id, team_id)` |
| `dim_player_season` | player x season | `(season_id, player_code)` |
| `fact_player_season_history` | player x prior season | `(player_code, season_id)` |
| `fact_gw_player` | player x fixture | `(season_id, gw, fpl_id, fixture_id)` |

Load order: `dim_season -> dim_player -> dim_team -> dim_player_season -> fact_player_season_history -> fact_gw_player`

### 4.2 Notable engineering decisions

**Stable cross-season player identity.** FPL resets `fpl_id` (element ID) every season. The pipeline bridges via `players_raw.code` (`player_code`) — the only stable identifier — rather than joining on `fpl_id` across seasons. This correctly resolves player histories for transfers, re-signed players, and name changes.

**Team keys are season-scoped surrogate keys.** `dim_team` uses a `(season_id, team_id)` composite rather than team name, since promoted/relegated teams recycle IDs across seasons.

**Double gameweek handling.** The fact table grain includes `fixture_id`, so DGW weeks correctly produce two rows per player rather than aggregated totals.

**`goals_conceded` is pitch-time scoped, not match-scoped.** In `fact_gw_player`, `goals_conceded` reflects goals conceded while the individual player was on the pitch — 75.9% of team-fixtures show inconsistent values across players on the same team. Team-level defensive stats are derived from `team_h_score` / `team_a_score` instead.

**Live data fetch.** `etl/fetch.py` calls three FPL API endpoints (bootstrap, fixtures, GW live) with exponential-backoff retry (3 attempts, base 2 s). It appends new GW rows to `merged_gw.csv` with deduplication on `(gw, fixture_id)` and overwrites `players_raw.csv` with the latest bootstrap snapshot.

### 4.3 Validation

Ten post-load assertion checks run automatically after each ETL cycle. Points reconciliation is scoped to completed seasons only — the live season is excluded because FPL retroactively corrects scores in `players_raw.csv` without back-propagating corrections to individual GW rows.

---

## 5. Exploratory Data Analysis

EDA was conducted across 32 logical checks and documented in `docs/eda_report.md`. Key findings that directly shaped modelling decisions:

**Era scoring drift.** Pre-xG seasons (1-6, 2016-17 to 2021-22) show a systematic -26.1% pts/GW relative to the xG era. This drift is structural — it results from FPL's redesigned scoring system and the absence of xG/xA/xGI features. Training on mixed-era data would introduce a confounding time trend. Decision: **xG era only** (seasons 7-10) for all modelling.

**Home advantage is position-dependent.** Home premium: GK +7.5%, DEF +18.7%, MID +11.1%, FWD +11.2%. `was_home` became a mandatory feature.

**Top-6 fixture penalty is large.** Facing a top-6 opponent reduces expected points: DEF -33.8%, FWD -21.2%, GK -17.6%, MID -16.6%. `opponent_season_rank` became mandatory.

**Team defensive strength explains 46.6% of GK/DEF variance.** `team_goals_conceded_season` (cumulative to date) became mandatory.

**Manager rows.** The 2024-25 season introduced FPL manager cards with `mng_*` columns populated and `value` as low as 0.5m. These rows are structurally different from player rows and are excluded via `WHERE mng_win IS NULL`.

---

## 6. Feature Engineering

Feature matrices are built per-position via `ml/features.py -> build_feature_matrix(position, era='xg')` and cached to parquet. The xG era base filter is:

```sql
WHERE mng_win IS NULL
  AND minutes > 0
  AND position_label IS NOT NULL
  AND season_gw_count >= 5
```

### 6.1 Feature categories

**Form features (rolling windows, within-player-season only):**
- `pts_rolling_3gw`, `pts_rolling_5gw` — recent form
- `minutes_rolling_5gw` — playing time stability
- `goals_rolling_5gw`, `assists_rolling_5gw` — attacking output (MID/FWD)
- `xg_rolling_5gw`, `xa_rolling_5gw` — expected contributions (MID/FWD)
- `saves_rolling_5gw`, `bonus_rolling_5gw` (lagged 1 GW)

Rolling windows are deliberately computed within `(player_code, season_id)` only — never chained across seasons or the 2019-20 COVID gap — to prevent season-boundary contamination.

**Season-to-date cumulative features:**
- `team_goals_conceded_season` — team's total goals conceded up to current GW
- `season_starts_rate_to_date` — fraction of possible GWs started (proxy for managerial trust)

**Fixture context:**
- `was_home` — binary home/away flag
- `opponent_season_rank` — opponent's current league position (derived via CTE from match results)
- `fixture_difficulty_rating` — FPL-issued 1-5 scale

**Player market features (lagged 1 GW to prevent leakage):**
- `transfers_in_lag1`, `transfers_out_lag1` — crowd sentiment
- `value` — price in FPL units

**Historical season features:**
- `prev_season_pts_per_gw` — last season's average (from `fact_player_season_history`)
- `career_seasons` — experience proxy

### 6.2 Leakage policy

The following columns are strictly banned as features:
- `bonus`, `bps`, `ict_index` — post-match computed; not available pre-GW
- Same-GW `clean_sheets`, `goals_scored`, `assists` — direct components of the target
- Same-GW `transfers_in`, `transfers_out`, `selected` — reactive signals available only after the GW resolves

---

## 7. Modelling Approach

### 7.1 Framework

Models are registered in `ml/models.py` via a `ModelSpec` dataclass carrying `build_fn`, `predict_fn`, `family`, `tier`, `requires_imputation`, `requires_scaling`, and `deps` (for meta-models). This registry pattern replaces per-model `if/elif` dispatch and enables the CV loop to iterate over any subset of the 22 registered models without structural changes.

### 7.2 CV strategy

**Expanding-window temporal CV — 3 folds:**

| Fold | Train | Validate |
|------|-------|----------|
| 1 | Season 7 | Season 8 |
| 2 | Seasons 7-8 | Season 9 |
| 3 | Seasons 7-9 | Season 10 |

Random train/test splits are explicitly prohibited. Temporal ordering is preserved by treating each season as a single block — a player's GW 38 data is never seen before their GW 1 data in any fold. The number of folds is computed dynamically from `etl/schema.py` `SEASONS` metadata; adding a new xG-era season automatically creates a fourth fold.

### 7.3 Preprocessing

**Ridge:** Stratified mean imputation (per `season_id`, training fold only) with a global fallback mean for unseen seasons at inference. `StandardScaler` fitted on training fold only, applied to validation fold. Neither the imputer nor scaler sees validation data at any point.

**LightGBM:** Native NaN handling; no imputation or scaling required.

### 7.4 Model inventory (22 models across 3 tiers)

**Tier 1 (primary):** Ridge, LightGBM — 8 position-specific models, 168 serialised artefacts total.

**Tier 2 (alternatives):** BayesianRidge, Lasso, ElasticNet, PoissonGLM, HuberRegressor, PolynomialRidge, QuantileRegressor, MLP, RandomForest, ExtraTreesRegressor, XGBoost.

**Tier 3 (meta-ensembles):** SimpleAverage, Stacking (linear meta-learner on OOF), Blending (Ridge + BayesianRidge + PoissonGLM + MLP weights).

**Sequential (separate pipeline):** LSTM and GRU — registered as stubs; full implementation in `ml/evaluate_sequential.py`; not serialised into production.

### 7.5 Ridge improvements (Phase 8)

`xgi_rolling_5gw` was dropped from Ridge for MID and FWD. The composite `(xG, xA, xGI)` feature set produced unexpected negative Ridge coefficients on `xGI` due to Ridge distributing weight across correlated predictors. Dropping `xGI` resolves this without sacrificing predictive power (xG + xA already encode the same signal).

An alpha grid search over `[0.1, 0.5, 1.0, 5.0, 10.0]` via 3-fold CV selected: GK=10.0, DEF=0.1, MID=0.1, FWD=0.1. GK's optimal alpha is 100x larger, consistent with its far smaller training set (2,731 rows vs. 13,723+ for outfield positions).

### 7.6 Uncertainty quantification

`BayesianRidge` outputs posterior predictive standard deviation per player via `model.predict(return_std=True)`. This is exposed in prediction CSVs as `pred_bayesian_ridge_std` and used in the dashboard as a per-player uncertainty signal for captain recommendations.

---

## 8. Model Evaluation

### 8.1 Metrics

| Metric | What it measures |
|--------|-----------------|
| MAE | Average absolute prediction error in FPL points — directly interpretable |
| RMSE | Error with larger penalties for big misses |
| Spearman rho | Rank-order correlation — relevant because FPL decisions are ranking decisions |
| Top-10 precision | Fraction of actual top-10 scorers captured in predicted top-10 — the operational metric |

### 8.2 Cross-validated performance — Ridge (production model)

| Position | Best alpha | CV MAE | CV RMSE | CV Spearman rho |
|----------|:----------:|:------:|:-------:|:---------------:|
| GK | 10.0 | 2.130 | 2.808 | 0.118 |
| DEF | 0.1 | 2.138 | 2.897 | 0.277 |
| MID | 0.1 | 1.835 | 2.514 | 0.371 |
| FWD | 0.1 | 2.270 | 3.022 | 0.413 |

**Baseline gate:** All 8 Tier-1 models (Ridge and LightGBM, all positions) beat the rolling-mean baseline on at least 2 of 3 primary metrics. 17 of 22 models pass the gate; the 3 consistent failures (FDR mean, Lasso, PolynomialRidge) have documented structural limitations.

**Ridge vs LightGBM:** Ridge outperforms LightGBM on every metric for all four positions with default LightGBM hyperparameters. LightGBM's advantages are expected to materialise after Optuna tuning — deferred, with FWD identified as highest priority.

**Blending ensemble:** Outperforms Ridge on GK (MAE 2.123 vs 2.132) and DEF (2.121 vs 2.138). Used alongside Ridge as a default output of the live pipeline.

### 8.3 Stratified analysis

**Home/away:** Ridge consistently performs better on home fixtures across all positions, consistent with the home-advantage signal captured in `was_home`.

**Opponent tier:** Top-6 fixture predictions have higher residual variance for DEF and FWD — the fixture difficulty signal is real but noisy at the individual match level.

**Price band and minutes bucket:** Low-playing-time players (< 45 min/GW average) are systematically harder to predict — benching decisions introduce irreducible noise. High-price premium players show moderate positive bias (the model underestimates their ceiling).

### 8.4 Production monitoring (live season)

**Alert thresholds (1.5x baseline MAE):** GK 3.494 | DEF 3.498 | MID 2.996 | FWD 3.609

**GW 30 (2025-26) results:**

| Position | GW MAE | Rolling 5-GW MAE | Status |
|----------|:------:|:----------------:|--------|
| GK | 1.988 | 2.053 | PASS |
| DEF | 2.735 | 2.491 | PASS |
| MID | 1.742 | 1.851 | PASS |
| FWD | 1.825 | 1.961 | PASS |

All positions are well within thresholds. GW 30 DEF MAE (2.74) is higher than average — inspection of the eval report shows multiple defenders scoring 11-12 points from defensive bonus events (clean sheets + saves cascade), which are inherently difficult to predict at the individual player level.

---

## 9. Dashboard / Application Layer

The dashboard is a 6-page Streamlit application served from `outputs/dashboards/app.py`, with pages in `pages/` and a shared utilities layer in `utils.py`.

**Launch:** `streamlit run outputs/dashboards/app.py` -> `http://localhost:8501`

| Page | Key content |
|------|-------------|
| Landing | Per-position MAE cards; top-10 predictions for latest GW |
| 1 — Data Explorer | Points distributions, home/away splits, team strength heatmap, player career trajectories, xG scatter, era comparison |
| 2 — Bias & Data Quality | ML bias audit (10 quantified biases), feature era availability, fixture difficulty effect, price vs performance |
| 3 — Model Performance | CV comparison table, OOF calibration scatter, MAE-by-fold, SHAP feature importance, residuals, learning curves, monitoring trend, per-GW eval report viewer |
| 4 — GW Predictions | FDR calendar, captain recommendation cards, filterable prediction table with uncertainty, ownership bubble chart, CSV export |
| 5 — Player Scouting | Boom/bust quadrant, value picks, form vs price, head-to-head player comparison, price trajectory, component analysis |
| 6 — Database Explorer | 20 preset SQL templates across 4 categories, table schema browser, free-form SQL editor |

**Engineering choices:**
- `@st.cache_data` decorates all data-loading functions in `utils.py` to avoid repeated DB queries across rerenders.
- The SQLite connection uses a read-only URI (`?mode=ro`) so dashboard users cannot accidentally modify the database.
- The FDR calendar is derived from `opponent_season_rank` in the parquet feature cache — not from `fact_gw_player` — so it remains available even when predictions are generated ahead of GW results.

---

## 10. Key Insights & Findings

1. **Era scoping is critical.** Training on all 10 seasons produces measurably worse models than restricting to the 4 xG-era seasons. The pre-xG data introduces a systematic -26.1% pts/GW confound that a regression model cannot cleanly separate from player-quality signal.

2. **Ridge outperforms tree models at this data scale.** With 2,731-19,495 training rows per position and a feature set of ~25-35 mostly continuous columns, Ridge's L2 regularisation is better calibrated to the signal-to-noise ratio than LightGBM's tree splits. This is expected to invert after Optuna tuning on LightGBM.

3. **Rank prediction is harder than point prediction.** Spearman rho values (0.12-0.41) are modest — the model is better at distinguishing broad tiers (blanks, low-scorers, high-scorers) than identifying the exact rank-1 player in a GW. This is consistent with the inherent randomness in football outcomes.

4. **Goalkeeper prediction is fundamentally harder.** GK MAE (2.130) is comparable to DEF (2.138) in absolute terms, but GK has only 2,731 training rows — roughly 20% of the DEF pool. The model generalises well given the data constraint, but Spearman rho (0.118 vs 0.277-0.413 for outfield) reveals limited rank discrimination.

5. **Defensive bonus events dominate GK/DEF volatility.** Clean sheets, saves bonuses, and goal involvement for defenders are episodic — they create high-scoring outliers that regression models systematically underpredict. GW 30 shows four defenders scoring 11-12 points with prediction errors of 7-10 points; all are clean-sheet + bonus combinations.

6. **Monitoring confirms stable in-season performance.** Both evaluated gameweeks (GW 24, GW 30) are comfortably within 1.5x baseline thresholds with rolling MAEs of 2.05 (GK), 2.49 (DEF), 1.85 (MID), 1.96 (FWD). There is no detectable model drift through GW 30.

---

## 11. Limitations

**Irreducible noise from team-sheet uncertainty.** Whether a player starts is unknown at prediction time. The model uses `season_starts_rate_to_date` as a proxy, but a player on 90% starts rate still has a non-trivial probability of being benched or substituted early. This is the single largest source of unpredictable error.

**Bonus point unpredictability.** Bonus points are allocated post-match by a BPS algorithm that cannot be reliably pre-calculated. Bonus is banned as a feature (leakage policy), but it accounts for a meaningful fraction of score variance in low-minute games.

**Thin training data for GK and FWD.** With 2,731 and 4,951 rows respectively, the GK and FWD models have limited capacity to learn fine-grained patterns. Three-fold CV provides only 745-1,644 training rows in Fold 1.

**No injury or suspension signal.** The pipeline does not currently ingest injury news or suspension data. A player flagged "doubtful" in the FPL system might still appear with a high predicted score if their recent form is strong.

**Single-season drift not yet observed.** Monitoring has only two data points (GW 24 and GW 30 of the same season). The retraining orchestrator exists but has not yet been run through a full season cycle. Its effectiveness is unvalidated.

**LightGBM not tuned.** Optuna hyperparameter tuning has been scoped and the code is implemented (`--tune` flag on `ml/train.py`) but has not yet been executed. Ridge may not remain the superior model after tuning.

---

## 12. Future Work

**Immediate (high-leverage, low-effort):**
- Run Optuna LightGBM tuning — `python -m ml.train --tune --position FWD` first. FWD has the most to gain based on current MAE gap.
- Add injury/availability signal: scrape FPL's `status` field (A/D/I/S/U) from the bootstrap API and include it as a feature or hard-override filter.
- Expand monitoring to include top-10 precision trends over time — MAE and RMSE alone do not fully capture rank-discrimination degradation.

**Medium-term:**
- Introduce a fixture difficulty residual model: a secondary model trained specifically on the prediction error as a function of `opponent_season_rank`, `was_home`, and season context. This could improve calibration for extreme fixture matchups.
- Backfill intermediate GWs programmatically rather than requiring manual `etl.fetch` calls.

**Longer-term:**
- Sequence modelling: LSTM/GRU pipelines are stubbed in `ml/evaluate_sequential.py`. Full integration would require careful handling of variable sequence lengths across season boundaries.
- Team formation and press coverage features via third-party expected goals APIs (Understat, FBref) for richer tactical context.
- Automated season-end retraining trigger: hook `retrain_season.py` to a post-season API signal rather than manual execution.

---

## 13. Conclusion

This pipeline demonstrates end-to-end applied data science practice: careful ETL with a principled schema, rigorous temporal validation that respects the sequential nature of the data, a modular model registry that separates algorithms from pipeline logic, and a production monitoring layer that closes the feedback loop after each gameweek.

The core result — a Ridge regression with CV MAE of 1.84-2.27 FPL points across four positions — is honest about the inherent difficulty of the prediction task. Football outcome variance is large; the model captures systematic patterns (form, fixtures, home advantage, team defensive strength) while acknowledging that individual gameweek outcomes contain substantial irreducible noise. The monitoring data through GW 30 confirms the model is stable in production.

The project is fully reproducible from the published release asset (`data.zip`) and source code, building the full database and running the dashboard in two commands.
