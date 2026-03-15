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
| 3 | EDA |  Partial |
| 4 | Feature Engineering |  Not started |
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

**What exists:** `logs/data_biases.md` — 10 biases quantified with SQL-backed findings,
including team strength confounding (r=−0.90), fixture difficulty gap (30–50% pts
difference), survivorship skew (75% of data is 30+ GW starters), home advantage
(+12–20% by position), and temporal drift (−26% pts/GW over 10 seasons).

**What to build — `eda/` directory:**

### 3.1 Target variable analysis
- Distribution of `total_points` per GW: histogram, box plot by position
- Proportion of blanks (0–1 pts), low scores (2–4), hauls (12+) by position
- Points per 90 minutes (filter `minutes > 0`): highlights true performance vs benching
- Season-level totals: distribution of `dim_player_season.total_points` by position/price band

### 3.2 Temporal analysis
- Average pts/GW by season (the −26% drift documented in data_biases.md)
- Era comparison: pre-xG (2016–22) vs xG era (2022–26) — pts distribution, feature availability
- GW-by-GW scoring patterns within seasons: early-season vs late-season effects
- COVID season (2019-20): GW gap at 30–38, confirm data integrity

### 3.3 Position analysis
- Per-position stat profiles: goals/90, assists/90, CS rate, bonus rate, ICT breakdown
- Cross-position points comparison — demonstrates why position-specific models are required
- MID sub-role heterogeneity: goals/90 distribution within MID (7x range identified)
- GK scoring drivers: CS rate vs saves rate vs team defensive quality

### 3.4 Team & fixture analysis
- Team CS rate league table per season (14.8x range between top and bottom)
- Home vs away points split by position (DEF +20.5%, others +10–12%)
- Top-6 vs rest fixture effect (+21–49% pts penalty when facing top-6)
- Correlation heatmap: `team_goals_conceded_season` vs DEF/GK pts

### 3.5 Player & price analysis
- `start_cost` vs `total_points` scatter with regression line (r=0.69)
- Price band performance table (£5m brackets)
- Career length distribution: 39% appear only 1 season, 13% appear 6+ seasons
- Minutes distribution: 28% of player-seasons have 0 minutes (never played)
- Survivorship: compare feature distributions for regular starters vs rotational players

### 3.6 Correlation & feature relevance
- Pearson/Spearman correlation matrix: all numeric features vs `total_points`
- Lag-1 correlation: does GW N−1 performance predict GW N? (validates rolling features)
- Missing data matrix: visual map of which columns are NULL in which seasons/eras

### 3.7 Outputs
- `eda/eda_report.ipynb` or `eda/eda_report.py` — runnable analysis
- `outputs/eda/` — exported charts (PNG/HTML)
- Findings feed directly into feature engineering decisions in Phase 4

---

## Phase 4 — Feature Engineering

**Entry point:** `ml/features.py` — SQL query layer that produces a clean ML-ready
DataFrame from `db/fpl.db`.

### 4.1 Base filter
```sql
WHERE mng_win IS NULL          -- exclude manager cards (2024-25)
  AND minutes > 0              -- exclude DNP appearances
  AND season_gw_count >= 5     -- exclude sparse player-seasons (< 5 GW appearances)
```

### 4.2 Era strategy decision (resolve before engineering)
- **Option A — xG era only (recommended):** seasons 7–10 (2022-23 to 2025-26), ~96,000 rows.
  Full feature set including xG/xA/xGI/xGC/starts. Cleanest model inputs.
- **Option B — all 10 seasons:** add `era_id` flag (1=pre-xG, 2=xG era); xG features
  NULL for 60% of rows. Requires imputation strategy or era-conditional models.

### 4.3 Mandatory engineered features (from bias analysis)

| Feature | Derivation | Addresses bias |
|---------|-----------|----------------|
| `opponent_season_rank` | Final league position 1–20 per season, joined via `opponent_team_sk` | Fixture difficulty (CRITICAL) |
| `team_goals_conceded_season` | `SUM(goals_conceded)` per team per season from `fact_gw_player` | Team strength confounding (CRITICAL) |
| `era_id` | 1 = pre-xG (seasons 1–6); 2 = xG era (seasons 7–10) | Temporal drift (HIGH) — only needed for Option B |

### 4.4 Leakage rules (strict)
- **Never use** same-GW `transfers_in`, `transfers_out`, or `selected` as features
- **Never use** `end_cost` or post-season aggregate stats
- **Always lag** `transfers_in` / `transfers_out` by exactly 1 GW before use
- **Target encoding** (if used): encode within CV folds only, never on full dataset

### 4.5 Feature catalogue

**Identity / context (non-predictive, used for grouping/CV):**
- `season_id`, `gw`, `player_code`, `position_code`, `team_sk`

**Fixture context:**
- `was_home` (BOOLEAN) — mandatory; +12–20% home effect
- `opponent_season_rank` (1–20) — mandatory; 30–50% pts penalty vs top-6
- `team_goals_conceded_season` — mandatory for DEF/GK models

**Player form (rolling — computed from prior GWs only):**
- `pts_rolling_3gw`, `pts_rolling_5gw` — recent form
- `mins_rolling_3gw` — availability / rotation signal
- `goals_rolling_5gw`, `assists_rolling_5gw` — attacking contribution
- `cs_rolling_5gw` — defensive form (DEF/GK)
- `bonus_rolling_5gw` — bonus point tendency

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
Not all features apply to all positions. Apply this filtering when building position models:

| Feature group | GK | DEF | MID | FWD |
|--------------|:--:|:---:|:---:|:---:|
| xGC / CS features | ✓ | ✓ | — | — |
| Goals / xG features | — | — | ✓ | ✓ |
| Saves features | ✓ | — | — | — |
| Bonus / BPS | ✓ | ✓ | ✓ | ✓ |

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
| **LightGBM** | Leaf-wise gradient boosting | Fastest on tabular data; excellent at this scale; SHAP support | Overfits small datasets (not an issue here) |
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
- **Folds:** up to 7 folds (seasons 3→4, 4→5, …, 9→10) given 10-season dataset.

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
- Home vs away
- Minutes bucket: starter (60+ mins), rotation (30–59), cameo (<30)
- Opponent tier: top-6 vs rest
- Era: pre-xG vs xG era (if training on all seasons)
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
- Model performance degrades for rotation/fringe players (training data is 75% elite starters)
- Defensive predictions remain partially confounded by team quality even after normalisation
- MID predictions have higher variance due to role heterogeneity (7x goals/90 range within MID)
- Pre-2022-23 predictions lack xG-based features (if using all-seasons strategy)
- No external injury or team-news data — biggest predictive gap not addressable from this dataset

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
- Bias summary table (from `logs/data_biases.md`)
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
- See `logs/data_biases.md` for full bias analysis and ML mitigation guidance
- See `schema_design.md` for full DDL and schema rationale
