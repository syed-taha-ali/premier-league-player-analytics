# FPL Analysis — Data Science Pipeline Plan

**Project:** Fantasy Premier League player performance prediction
**Database:** `db/fpl.db` — 242,316 GW rows, 10 seasons (2016-17 to 2025-26)
**Goal:** Predict GW-level FPL points per player; surface insights via interactive dashboard

---

## Pipeline Overview

| # | Phase | Status |
|---|-------|--------|
| 1 | Data Collection |  Complete |
| 2 | Data Cleaning |  Complete |
| 3 | EDA |  Complete |
| 4 | Feature Engineering |  Complete |
| 5 | Modelling |  Not started |
| 6 | Evaluation |  Not started |
| 7 | Interactive Dashboard & Visualisations |  Not started |
| 8 | Deployment |  Not started |
| 9 | Monitoring |  Not started |

---

## Phase 1 — Data Collection 

**Sources ingested:**

| File | Role |
|------|------|
| `{season}/gws/merged_gw.csv` | Primary GW-level fact source (one row per player per fixture) |
| `{season}/players_raw.csv` | Player + season dimension data |
| `{season}/players/{name}/history.csv` | Authoritative start_cost / end_cost per season |
| `data/master_team_list.csv` | Team ID → name mapping (2016-17 to 2023-24) |

**Scope:** 10 seasons, ~13,900 CSV files, ~242,000 player-GW observations.

---

## Phase 2 — Data Cleaning 

**Output:** `db/fpl.db` — 6-table SQLite star schema, PRAGMA-tuned (WAL mode, 64 MB cache).

| Table | Grain | Rows |
|-------|-------|-----:|
| `dim_season` | season | 10 |
| `dim_player` | player (cross-season) | ~2,620 |
| `dim_team` | team × season | 200 |
| `dim_player_season` | player × season | 7,334 |
| `fact_gw_player` | player × fixture | 242,316 |
| `fact_player_season_history` | player × prior season | 5,419 |

**Validation:** 11 automated checks (all passing); 32 logical checks + 11 case studies in `logs/`.

---

## Phase 3 — EDA

**Status: Complete.** All six analysis sections implemented and documented.

**Deliverables:**
- `eda/eda_report.ipynb` — runnable Jupyter notebook (25 code cells, 30 markdown cells)
- `outputs/eda/` — 24 exported PNG charts
- `docs/eda_report.md` — full written report with chart interpretations (Section 1) and
  Phase 4 implications and recommendations (Section 2)
- `docs/data_biases.md` — 10 biases quantified with SQL-backed findings

**Key findings feeding Phase 4:**

### 3.1 Target variable analysis
- All positions share heavy right-skew: >70% of GW appearances score 0–1 pts; hauls (<2%) drive prediction variance
- Means: GK 0.99, DEF 1.20, MID 1.33, FWD 1.40; standard deviations exceed means (GW pts highly volatile)
- Points per 90 (minutes > 0): FWDs lead at median ~4 pts/90; GK distribution most symmetric

### 3.2 Temporal analysis
- −26.1% pts/GW drift confirmed: peak 1.42 (2018-19) → trough 1.05 (2023-24)
- Pre-xG era mean 1.34 vs xG era mean 1.14 — 15% gap with incompatible feature sets → Option A adopted
- Early-season premium: GW 1–6 avg 1.375 vs GW 33–38 avg 1.154 (+16.0%)
- COVID 2019-20: GW gap at 30–38 confirmed clean; rolling features must not chain across GW 29 → 39

### 3.3 Position analysis
- GK scoring driven almost entirely by CS rate (r = 0.795); saves show weak negative correlation (r = −0.12)
- MID goals/90 coefficient of variation = 0.932 vs FWD 0.490 — sub-role heterogeneity is structural
- Per-position profiles confirm fully non-overlapping feature sets across positions

### 3.4 Team and fixture analysis
- Team CS rate range: Arsenal 0.474 vs Sheffield United 0.027 — 17.8x range (2023-24)
- Home premium (confirmed across all 10 seasons): GK +7.5%, DEF +18.7%, MID +11.1%, FWD +11.2%
- Top-6 fixture penalty: GK −17.6%, DEF −33.8%, MID −16.6%, FWD −21.2%
- Team goals conceded explains 46.6% of variance in DEF/GK average GW points (r = −0.683)

### 3.5 Player and price analysis
- start_cost vs season_total_points: overall r = 0.505; per position 0.467–0.603
- 39.4% of players appear in only one season — cold-start problem for rolling features
- 27.6% of player-seasons have zero minutes (DNP); 21.4% are regular starters (2000+ min)

### 3.6 Correlation and feature relevance
- Top same-GW correlates with `total_points`: `bonus` r ≈ 0.74, `bps` r ≈ 0.70, `ict_index` r ≈ 0.65 — all post-match leakage; banned as features
- Genuinely predictive signals: `minutes` r ≈ 0.55, `expected_goal_involvements` r ≈ 0.48 (MID/FWD), `clean_sheets` r ≈ 0.55 (GK/DEF — rolling lag only)
- Lag-1 autocorrelation: Pearson r = 0.378, Spearman rho = 0.650 over 234,686 GW pairs — validates rolling-window features

---

## Phase 4 — Feature Engineering

**Entry point:** `ml/features.py` — SQL query layer that produces a clean ML-ready
DataFrame from `db/fpl.db`.

### 4.1 Base filter
```sql
WHERE mng_win IS NULL          -- exclude 322 manager rows (2024-25)
  AND minutes > 0              -- exclude 27.6% DNP appearances
  AND position_label IS NOT NULL
  AND season_gw_count >= 5     -- exclude sparse player-seasons (< 5 GW appearances)
```

### 4.2 Era strategy — Option A adopted

**Option A — xG era only (seasons 7–10, 2022-23 to 2025-26), ~96,000 rows.**

Adopted based on EDA findings:
1. Pre-xG seasons lack the most predictive attacking features (`xG/xA/xGI/xGC`), which are absent for 60% of historical rows.
2. The −26.1% scoring drift (peak 1.42 pts/GW in 2018-19 → trough 1.05 in 2023-24) means pre-2022-23 data represents a different outcome distribution requiring explicit normalisation.
3. The xG era alone (~96,000 filtered rows) is sufficient for position-specific LightGBM and Ridge models with 3-fold expanding-window CV.
4. Option B adds substantial engineering complexity (era flags, imputation, era-conditional feature paths) for uncertain marginal gain.

If Option B is pursued in future: mandatory additions are `era_id` flag (1=pre-xG, 2=xG), season-mean-normalised `total_points` as training target, and xG features imputed as zero (not NaN) with an accompanying `has_xg` boolean indicator.

### 4.3 Mandatory engineered features (EDA-confirmed)

| Feature | Derivation | EDA justification |
|---------|-----------|-------------------|
| `opponent_season_rank` | Final league position 1–20 per season, joined via `opponent_team_sk → dim_team → (season_id, team_id)` | Top-6 penalty: DEF −33.8%, FWD −21.2%, GK −17.6%, MID −16.6%. A static top-6 flag under-captures season-specific difficulty. (CRITICAL) |
| `team_goals_conceded_season` | `SUM(goals_conceded)` per `(team_sk, season_id)` from `fact_gw_player`, lagged to exclude current GW | 46.6% of variance in DEF/GK avg pts explained by this single variable (team-season level r = −0.683; player-level r = −0.90 per biases analysis). (CRITICAL) |
| `was_home` | Already present in `fact_gw_player` — no derivation needed | Home premium confirmed: GK +7.5%, DEF +18.7%, MID +11.1%, FWD +11.2%, consistent across all 10 seasons and validated by COVID neutral-venue natural experiment. (MANDATORY) |

### 4.4 Leakage rules (strict)

**Post-match computed — never use as features:**
- `bonus`, `bps`, `ict_index` — highest same-GW correlates with `total_points` (r ≈ 0.74, 0.70, 0.65) but all are computed post-match. Use only their lagged or rolling counterparts.

**Same-GW target components — never use as features:**
- `clean_sheets`, `goals_scored`, `assists` — direct components of `total_points`. Use only their rolling lags as form signals.

**Crowd-signal leakage — lag before use:**
- `transfers_in`, `transfers_out`, `selected` — same-GW transfer activity is reactive (35x spike after 15+ pt GWs per biases analysis). Lag by exactly 1 GW if included.

**Post-season / future data — never use:**
- `end_cost` or any post-season aggregate stats
- **Target encoding** (if used): encode within CV folds only, never on full dataset

### 4.5 Feature catalogue

**Identity / context (non-predictive, used for grouping/CV):**
- `season_id`, `gw`, `player_code`, `position_code`, `team_sk`

**Fixture context:**
- `was_home` (BOOLEAN) — mandatory; EDA-confirmed: GK +7.5%, DEF +18.7%, MID +11.1%, FWD +11.2%
- `opponent_season_rank` (1–20) — mandatory; EDA-confirmed: DEF −33.8%, FWD −21.2%, GK −17.6%, MID −16.6% penalty vs top-6
- `team_goals_conceded_season` — mandatory for DEF/GK models; explains 46.6% of variance in defensive player scoring

**Player form (rolling — computed from prior GWs only):**
- `pts_rolling_3gw`, `pts_rolling_5gw` — primary form indicators (lag-1 Pearson r = 0.378, Spearman rho = 0.650)
- `mins_rolling_3gw` — availability / rotation signal (more predictive than pts in some cases)
- `goals_rolling_5gw`, `assists_rolling_5gw` — attacking contribution
- `cs_rolling_5gw` — defensive form (DEF/GK only)
- `bonus_rolling_5gw` — bonus point tendency (FWDs: highest bonus rate at 0.352/app)
- `saves_rolling_5gw` — GK only; rotation proxy (GK on pitch = positive saves count)

**Rolling boundary rule:** Roll within `(player_code, season_id)` only. Never chain rolling features across season boundaries or across the 2019-20 COVID GW gap (season_id = 4, GW 29 → 39).

**Season-to-date (cumulative — no leakage, always based on prior GWs):**
- `season_pts_per_gw_to_date` — season average so far
- `season_starts_rate_to_date` — proportion of GWs started (2022-23+)

**Static player/season attributes:**
- `start_cost` (÷10 for £m) — prior-season perceived quality signal
- `value` lagged 1 GW — in-season price momentum
- `transfers_in_lag1`, `transfers_out_lag1` — crowd-wisdom signal

**xG-based (xG era only, seasons 7–10):**
- `xg_rolling_5gw`, `xa_rolling_5gw` — expected contribution
- `xgi_rolling_5gw` — combined attacking threat
- `xgc_rolling_5gw` — expected goals conceded (DEF/GK)

**Team form:**
- `team_pts_rolling_3gw` — aggregate team performance
- `team_cs_rolling_3gw` — team defensive form (DEF/GK)
- `team_goals_scored_rolling_3gw` — attacking team context (FWD/MID)

**Opponent features:**
- `opponent_goals_scored_season` — proxy for opponent attacking threat
- `opponent_cs_rate_season` — proxy for opponent defensive strength

### 4.6 Position-specific feature subsets

Feature applicability by position:

| Feature group | GK | DEF | MID | FWD |
|--------------|:--:|:---:|:---:|:---:|
| xGC / CS / team_goals_conceded features | ✓ | ✓ | — | — |
| Goals / xG / xGI features | — | — | ✓ | ✓ |
| Saves / saves_rolling_5gw | ✓ | — | — | — |
| Bonus rolling features | ✓ | ✓ | ✓ | ✓ |
| opponent_season_rank / was_home | ✓ | ✓ | ✓ | ✓ |

**Per-position guidance (EDA-confirmed):**

**GK model:**
- `team_cs_rolling_3gw` is the primary form signal — GK performance is a team-quality proxy (CS rate r = 0.795 with avg GW pts)
- `team_goals_conceded_season` mandatory (team-season r = −0.683)
- `saves_rolling_5gw` — weak individual predictor (r = −0.12) but useful as rotation indicator
- Exclude `expected_goals` and `expected_assists` (near-zero for GKs)

**DEF model:**
- `team_goals_conceded_season` is the single most important feature (46.6% variance explained)
- `opponent_season_rank` provides the fixture difficulty adjustment — most impactful for DEFs (−33.8% penalty vs top-6)
- `was_home` — +18.7% home premium is the largest of any position
- `cs_rolling_5gw` for individual defensive form

**MID model:**
- `xgi_rolling_5gw` (`expected_goal_involvements`) is the primary attacking signal (xG era)
- Be aware: CV = 0.932 within MID vs 0.490 for FWD — higher residual error is structural, not a modelling failure
- Consider `goals_scored / (goals_scored + assists + 0.01)` as a goal-vs-assist ratio feature to implicitly distinguish sub-roles (striker-like vs creator midfielders)

**FWD model:**
- `xg_rolling_5gw` is the primary signal
- `bonus_rolling_5gw` — FWDs have the highest bonus rate (0.352/app) and bonus tends to be sticky for top scorers

### 4.7 Output
- `ml/features.py` — `build_feature_matrix(position, era='xg')` → returns clean DataFrame
- `outputs/features/feature_matrix_{position}.parquet` — cached for model training

---

## Phase 5 — Modelling

**Target:** `total_points` (GW-level regression, continuous output)
**Architecture:** Four position-specific models — GK, DEF, MID, FWD. Never cross-position.
**Validation:** Expanding-window temporal CV (train on seasons N…k, test on k+1). Never random split.

---

### 5.1 Full Model Brainstorm

Every model type that could plausibly be applied to this dataset, with honest assessment.

#### A — Naive Baselines (always implement — benchmark floor)

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **Position mean** | Predict mean pts by position × home/away | Zero effort; interpretable | Ignores all player-level signal |
| **Rolling N-GW mean** | Player's average pts over last 3 or 5 GWs | Captures form; fast | Ignores fixture difficulty |
| **FDR-adjusted mean** | Rolling mean × opponent difficulty multiplier | Adds fixture context | Still ignores xG, price, team |
| **Last season avg** | Player's pts/GW from prior season | Reasonable for season openers | Stale after a few GWs |

#### B — Linear Models

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **OLS Linear Regression** | Standard least-squares | Fully interpretable; fast | Assumes linearity; no regularisation |
| **Ridge Regression** | OLS + L2 penalty | Handles correlated features well; stable | Still linear |
| **Lasso Regression** | OLS + L1 penalty | Automatic feature selection | Unstable with correlated features |
| **ElasticNet** | L1 + L2 combined | Best of Ridge + Lasso | Extra hyperparameter |
| **Polynomial + Ridge** | Degree-2 feature interactions + Ridge | Cheap non-linearity | Feature explosion; harder to interpret |
| **Poisson GLM** | GLM with Poisson link (log scale) | Natural for count-like pts distribution | Assumes Poisson variance = mean |

#### C — Tree-Based Models

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **Decision Tree** | Single CART tree | Fully interpretable; visualisable | High variance; overfits easily |
| **Random Forest** | Bagging of decision trees | Robust; good OOB estimate; handles NULLs | Slow to train; high memory |
| **Extra Trees** | Random splits + bagging | Faster than RF; lower variance | Slightly less accurate |

#### D — Gradient Boosting (primary candidates)

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **XGBoost** | Gradient boosting with regularisation | Excellent performance; native NULL handling; SHAP support | Slower than LightGBM |
| **LightGBM** | Leaf-wise gradient boosting | Fastest on tabular data; excellent at this scale; SHAP support | Overfits small datasets — not an issue for DEF/MID, but GK Fold 1 has only 745 training rows; use conservative hyperparameters for GK (see §5.3) |
| **CatBoost** | Gradient boosting with native categorical support | Handles `position`, `team`, `era_id` without encoding | Slower than LightGBM; less common |
| **HistGradientBoosting** | sklearn's GB with histogram binning | Pure sklearn; no extra dependencies | Less configurable than XGB/LGBM |

#### E — Neural Networks

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **MLP (Multi-Layer Perceptron)** | Fully-connected feed-forward net | Flexible; learns feature interactions | Needs more tuning; less interpretable |
| **LSTM** | Recurrent net over GW sequences | Captures sequential GW dependencies natively | Needs fixed-length sequences; slow; more data needed |
| **GRU** | Lighter variant of LSTM | Faster than LSTM; similar performance | Same sequence-length requirement |
| **Temporal Fusion Transformer (TFT)** | Attention-based time-series model | State-of-art for multi-horizon TS forecasting; handles static + dynamic features | High complexity; significant engineering overhead |
| **N-BEATS / N-HiTS** | Neural basis expansion for TS | Strong TS benchmark models | Pure TS — harder to incorporate player-level static features |

#### F — Probabilistic / Bayesian Models

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **Bayesian Ridge** | Ridge with Bayesian priors | Uncertainty estimates per prediction | Marginal over deterministic Ridge |
| **Gaussian Process Regression** | Non-parametric Bayesian | Full predictive distribution | O(n³) — infeasible at 242K rows |
| **Zero-Inflated Poisson** | Poisson with excess-zero component | Models the many-zero scores naturally | Complex; limited library support |

#### G — Decomposed / Component Models

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **Goals + Assists + CS + Bonus separately** | One model per scoring component → sum to pts | Each component is more predictable; interpretable; position-natural | Covariance between components ignored; 4–5× model count |
| **Minutes model first** | Predict P(starts), then conditional pts | Directly models rotation risk (huge FPL factor) | Requires calibrated probability output |

#### H — Ensemble / Stacking

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **Simple Voting / Averaging** | Average predictions of multiple models | Easy; often beats any single model | Need diversity in base models |
| **Stacking (meta-learner)** | Train meta-model on OOF predictions | Can learn when each base model excels | Leakage risk if CV not done carefully |
| **Blending** | Average on a held-out set | Simpler than stacking | Wastes a holdout split |

#### I — Time-Series Specific (not recommended)

| Model | Reason not recommended |
|-------|----------------------|
| **ARIMA / SARIMA** | Max 38 GWs per player per season — too few points for reliable ARIMA fitting |
| **Prophet** | Designed for daily/weekly data with trend + seasonality — FPL has neither |
| **Exponential Smoothing (ETS)** | Better than ARIMA but same data-length problem |

---

### 5.2 Tiered Implementation Recommendations

#### Tier 1 — Implement First (highest ROI, low effort)

These should be built before any other model. They establish the benchmark and are the
most likely to be the final production models.

| Model | Rationale |
|-------|-----------|
| **Rolling mean baseline** | Zero-effort benchmark; any model that doesn't beat this is useless |
| **Ridge Regression** | Fast, interpretable, good at quantifying linear feature effects; essential before going non-linear |
| **LightGBM** | Best expected performance/effort ratio on tabular FPL data; native NULL handling; fast hyperparameter search; SHAP explainability |

#### Tier 2 — Implement After Tier 1 (meaningful additional value)

Build these once Tier 1 is working and evaluated. Each adds a distinct perspective.

| Model | Rationale |
|-------|-----------|
| **XGBoost** | Close competitor to LightGBM; different regularisation; worth comparing directly |
| **Random Forest** | Different inductive bias from boosting; often catches different patterns |
| **Decomposed model (minutes → pts)** | Directly models the biggest FPL risk (rotation/injury); architecturally novel |
| **MLP** | Neural baseline; validates whether deep learning adds value over boosting |

#### Tier 3 — Experimental (high effort, uncertain marginal gain)

Only pursue after Tier 1+2 are solid and evaluated.

| Model | Rationale |
|-------|-----------|
| **LSTM / GRU** | Worth testing if EDA shows strong sequential GW dependencies that rolling features don't capture |
| **Temporal Fusion Transformer** | State-of-art TS model; justified only if LSTM shows strong improvement over LightGBM |
| **Stacking ensemble** | Blends Tier 1+2 models; typically adds 1–3% MAE improvement if base models are diverse |
| **Decomposed component models (goals/assists/CS separately)** | High interpretability payoff; test if position-specific pts breakdown reveals better signal |

#### Not recommended for this dataset

| Model | Reason |
|-------|--------|
| ARIMA/SARIMA/Prophet | Too few per-player GWs; not designed for cross-sectional player panels |
| Gaussian Process | O(n³) — infeasible at scale |
| Zero-Inflated Poisson | Marginal over simpler approaches; complex implementation |

---

### 5.3 Training setup

**File structure:**
```
ml/
├── features.py          # build_feature_matrix(position, era)
├── train.py             # train and serialise models
├── evaluate.py          # CV evaluation + metrics
└── predict.py           # inference on new GW fixture list
models/
└── {position}_{model}.pkl
```

**NaN handling:** Feature matrices contain ~4.5% NaN in player rolling features (first GW
of each player-season) and ~2.8% NaN in team/opponent features (first fixture of each
team-season). LightGBM handles NaN natively — no action needed. Ridge requires imputation
before `fit()`; use mean imputation computed within the training fold only, stratified by
`(position, season_id)` to avoid contaminating early-season rows with mid-season averages.
Never fit the imputer on the full dataset or on the validation fold.

**Feature scaling:** Ridge requires standardisation; LightGBM does not. Fit a
`StandardScaler` on the training fold only and apply the same fitted scaler to the
validation fold. Do not refit on the validation fold.

**Position-specific LightGBM starting hyperparameters:**

| Position | `num_leaves` | `min_child_samples` | `learning_rate` | `n_estimators` | Rationale |
|----------|:------------:|:-------------------:|:---------------:|:--------------:|-----------|
| GK | 15 | 30 | 0.05 | 200 | Only 745 training rows in Fold 1 — aggressive regularisation required |
| DEF | 31 | 20 | 0.05 | 300 | Comfortable row count; standard regularisation |
| MID | 31 | 20 | 0.05 | 300 | Largest position; standard regularisation |
| FWD | 31 | 20 | 0.05 | 300 | Fold 1 is marginal (1,459 rows) but viable at standard settings |

These are starting points for hyperparameter search, not fixed values.

**Hyperparameter tuning:** Optuna (Bayesian search) or sklearn GridSearchCV within each
CV fold. Tune on validation fold; do not touch test fold.

**Reproducibility:** Set `random_state=42` everywhere; log all hyperparameters to
`logs/training/`.

---

## Phase 6 — Evaluation

### 6.1 Validation framework
- **Method:** Expanding-window temporal CV — train on seasons 1…k, validate on k+1.
  Minimum training window: 2 seasons. Test window: 1 season.
- **Never:** random train/test split (would leak future into training).
- **Folds:** 3 folds within the xG era (season 7→8, seasons 7-8→9, seasons 7-9→10).
  Rationale: Option A adopted (xG era only); the pre-xG seasons are excluded from training,
  so cross-era folds are not applicable.

### 6.2 Primary metrics (per position, per model, per CV fold)

| Metric | Unit | Why |
|--------|------|-----|
| **MAE** (Mean Absolute Error) | points | Interpretable in FPL units — "off by X pts on average" |
| **RMSE** | points | Penalises large mispredictions (missed hauls) more heavily |
| **R²** | — | Proportion of variance explained |
| **Spearman ρ** (rank correlation) | — | FPL is about ranking players, not absolute accuracy |
| **Top-N precision** | % | What % of predicted top-N scorers actually score top-N? |

### 6.3 Stratified evaluation
All metrics computed separately for:
- Position (GK / DEF / MID / FWD)
- Home vs away — EDA-confirmed effects large enough to distort aggregate metrics if not disaggregated
- Minutes bucket: starter (60+ mins), rotation (30–59), cameo (<30)
- Opponent tier: top-6 vs rest — EDA-confirmed −17% to −34% penalty by position
- Price band (£5m brackets)

### 6.4 Calibration & diagnostics
- **Calibration plot:** predicted vs actual mean pts in 10 equal-width bins — check for
  systematic over/under-prediction
- **Residual analysis:** residuals vs predicted, vs GW, vs opponent rank — identify
  structural blind spots
- **SHAP summary plots:** feature importance for LightGBM/XGBoost models per position
- **Learning curves:** training MAE vs validation MAE as training data grows — diagnose
  over/underfitting

### 6.5 Benchmark comparison table
Every model evaluated against the rolling-mean baseline. Report: MAE, RMSE, Spearman ρ,
top-10 precision. A model must beat the baseline on at least 2 of 3 primary metrics to
be considered for production.

### 6.6 Known limitations to document

| Limitation | Quantified impact | Mitigation status |
|------------|:-----------------:|-------------------|
| MID sub-role heterogeneity | CV = 0.932 within MID vs 0.490 for FWD | Partially mitigated by xGI rolling features; residual variance is structural |
| Survivorship bias | 75% of data from 30+ GW starters | Acknowledge degraded performance for rotation and fringe players in evaluation |
| Cold-start players (no prior-season history) | 39.4% of players appear only 1 season | Fall back to `start_cost` + position priors; rolling features unavailable until GW 3+ |
| Team quality confounding (DEF/GK) | 46.6% variance in DEF/GK pts explained by team goals conceded alone | Mitigated by `team_goals_conceded_season`; residual individual-skill signal remains weak |
| No injury or team-news data | Largest unaddressable predictive gap | Document as primary model limitation; not fixable from this dataset |
| 2019-20 COVID GW gap | GW 29 → 39 discontinuity in season_id = 4 | Enforce rolling boundary rule: no cross-gap feature chaining |

---

## Phase 7 — Interactive Dashboard & Visualisations

### 7.1 Architecture
- **Framework:** Streamlit (lower barrier; faster to iterate) or Plotly Dash (more
  control for production). Decide after Phase 6.
- **Static charts:** matplotlib / seaborn exported to `outputs/eda/`
- **Serving:** local (`streamlit run app.py`); no cloud deployment in initial phase
- **Output directory:** `outputs/dashboards/`

### 7.2 Dashboard pages / sections

#### Page 1 — Data Explorer (EDA insights)
- Season selector, position filter, era toggle
- Points distribution histogram (by position/season)
- Home vs away comparison bar chart (by position)
- Team strength heatmap: goals conceded per team per season
- Player search: career trajectory chart (pts/GW over all seasons)

#### Page 2 — Bias & Data Quality
- Bias summary table (from `docs/data_biases.md`)
- Missing data matrix: feature availability by season/era
- Fixture difficulty effect: pts by opponent rank scatter
- Price vs performance scatter (start_cost vs season_total_points)

#### Page 3 — Model Performance
- Model comparison table: MAE / RMSE / Spearman ρ per model × position
- Calibration plots: predicted vs actual (4 position subplots)
- Feature importance (SHAP bar chart, per position, per model)
- Residual plot: residuals vs GW, coloured by position

#### Page 4 — GW Predictions
- Fixture list input for next GW
- Ranked player predictions table: player, position, team, opponent, predicted pts,
  uncertainty band (if probabilistic model)
- Filter by position, price band, ownership %
- Download as CSV button

### 7.3 Static report charts (always export)
- `outputs/eda/points_distribution.png`
- `outputs/eda/home_away_effect.png`
- `outputs/eda/team_strength_heatmap.png`
- `outputs/models/calibration_{position}.png`
- `outputs/models/shap_importance_{position}.png`

---

## Phase 8 — Deployment

### 8.1 Model serialisation
- Serialise all production models with `joblib`: `models/{position}_{model}.pkl`
- Alongside each model, save: feature list, scaler (if used), training metadata
  (season range, n_rows, CV MAE) as `models/{position}_{model}_meta.json`
- Version models by season: `models/v{season_id}/` — never overwrite prior season models

### 8.2 Prediction pipeline
**Entry point:** `ml/predict.py`

**Inputs:**
- Next GW number and season
- Fixture list: `[(home_team, away_team), …]`
- Optional: current player prices and ownership (from FPL API or manual CSV)

**Process:**
1. Query `db/fpl.db` for all eligible players (minutes > 0, mng_win IS NULL)
2. Build feature matrix for next GW using `features.py` (rolling stats to date)
3. Attach fixture context: `was_home`, `opponent_season_rank`
4. Load serialised model for each position
5. Generate predictions; combine into single ranked DataFrame

**Output:**
- `outputs/predictions/gw{N}_predictions.csv` — columns: player, position, team,
  opponent, predicted_pts, model
- Console summary: top-5 predicted per position

### 8.3 Retraining cadence
- **End of season:** Retrain all models with new season appended to training data.
  Run full CV on new fold (new season as held-out test).
- **Mid-season trigger:** If rolling MAE (last 5 GWs) exceeds 1.5× baseline MAE,
  flag for review. Retrain if confirmed performance degradation.

---

## Phase 9 — Monitoring

### 9.1 Per-GW performance tracking
After each GW result is published:
1. Join `gw{N}_predictions.csv` against actual `fact_gw_player` results for that GW
2. Compute: MAE, RMSE, Spearman ρ, top-10 precision for that GW
3. Append to `logs/monitoring/monitoring_log.csv`

### 9.2 Rolling metrics
- Compute 5-GW rolling MAE per position per model
- Compare against baseline (rolling mean) rolling MAE
- Flag if model MAE > 1.5× baseline for any position over 5 consecutive GWs

### 9.3 Schema change alerting
FPL has added new column groups each season (xG in 2022-23, manager mode in 2024-25,
defensive stats in 2025-26). Before each season's data is loaded:
- Check `merged_gw.csv` column set against `etl/schema.py` era definitions
- Alert if new columns appear or existing columns are dropped
- Update `etl/schema.py`, `ml/features.py`, and `project_plan.md` accordingly

### 9.4 Output
- `logs/monitoring/monitoring_log.csv` — per-GW metrics
- `logs/monitoring/gw{N}_eval.md` — narrative summary each GW

---

## Project Conventions

- `data/` is **read-only** — all outputs go to `db/`, `models/`, `outputs/`, `logs/`
- All costs stored in **£0.1m units** in the DB — divide by 10 only in viz/reporting layer
- Python for all code; SQLite for data storage
- Validate each ETL stage before proceeding: `python -m etl.run`
- Never join on `fpl_id` across seasons — always bridge via `player_code`
- See `docs/data_biases.md` for full bias analysis and ML mitigation guidance
- See `docs/eda_report.md` for EDA findings and Phase 4 recommendations (Section 2)
- See `schema_design.md` for full DDL and schema rationale
