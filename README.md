# FPL Analysis — End-to-End Fantasy Premier League Analytics Pipeline

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/syed-taha-ali/premier-league-player-analytics)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-8.5%2F10-brightgreen)](docs/code_review.md)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

A complete data science pipeline for Fantasy Premier League: raw API data → validated
star-schema database → 21 trained models per position → interactive Streamlit dashboard.

---

## Overview

This project builds an end-to-end analytics system for Fantasy Premier League (FPL), the
official Premier League fantasy game with over 10 million active players. The goal is to
predict each player's gameweek points score to support transfer and captaincy decisions.

The pipeline covers 10 seasons (2016-17 to 2025-26) and 247,308 gameweek-level player rows
stored in a validated SQLite star schema. A live post-GW runner fetches the latest FPL API
data, rebuilds the database, generates predictions, and logs performance metrics against
actuals — all in a single command.

**Production model CV MAE (Ridge):** GK 2.130 | DEF 2.138 | MID 1.830 | FWD 2.270

---

## Key Features

**Data pipeline**
- FPL API client with exponential-backoff retry; appends new GW rows without overwriting historical data
- Full drop-and-rebuild ETL with 10 post-load validation checks: orphan foreign keys,
  row counts, season-level points reconciliation, GW range, NULL profiles by schema era
- Schema change alerting: non-fatally detects added or dropped API columns between seasons
- 6-table star schema with proper FK constraints, indices, and era-aware NULL profiles

**Feature engineering**
- 20 position-specific features: rolling attacking/defensive stats (3-GW and 5-GW windows),
  xG/xA/xGC, season-to-date averages, fixture difficulty via live opponent rank CTE,
  lagged value and transfer signals
- `shift(1)` applied before every rolling and expanding operation — zero same-GW leakage
- Opponent rank derived from a season league table CTE computed directly in SQLite

**Modelling**
- 21 models per position × 4 positions = 84 serialised bundles (+ 84 metadata files)
- Expanding-window temporal CV across the xG era (seasons 7–10); never a random split
- Automated baseline gate: every model benchmarked against a rolling-mean predictor
- Meta-models (Simple Average, Stacking, Blending) trained on out-of-fold predictions

**Live GW runner**
- `run_gw.py` — four-step pipeline: Fetch → ETL → Predict → Monitor
- Per-GW narrative eval reports: MAE, RMSE, Spearman ρ, top-10 precision, 5-GW rolling
  MAE with calibrated alert thresholds per position
- `retrain_season.py` — 9-step end-of-season orchestrator with dry-run preview

**Dashboard**
- 6-page Streamlit app with read-only SQLite access and full `@st.cache_data` caching
- Ranked GW predictions with fixture difficulty calendar and BayesianRidge uncertainty
- Player scouting with boom/bust profiling and season-by-season career stats
- 20 preset SQL templates across four query categories plus a free-form SQL editor

---

## Tech Stack

| Category | Libraries / Tools |
|---|---|
| Language | Python 3.x |
| Storage | SQLite (`sqlite3`), Parquet (`pyarrow`) |
| Data | pandas 2.3, numpy 2.4 |
| Statistics | scipy 1.17, statsmodels 0.14 |
| ML | scikit-learn 1.8, LightGBM 4.6, XGBoost 3.2 |
| Deep learning | PyTorch 2.10 (LSTM/GRU — evaluated separately) |
| Tuning | Optuna 4.8 |
| Explainability | SHAP 0.51 |
| Serialisation | joblib 1.5 |
| Dashboard | Streamlit ≥ 1.35, Plotly ≥ 5.20 |
| API client | requests 2.32 |
| Visualisation | matplotlib 3.10, seaborn 0.13 |

---

## Data Pipeline

```
FPL API
  │
  ├── etl/fetch.py        pulls bootstrap, fixtures, live GW stats
  │                       schema check runs here (non-fatal; logs alerts)
  │
  ├── etl/run.py          drops and rebuilds db/fpl.db from CSVs (~16s)
  │                       runs 10 post-load validation checks
  │
  ├── db/fpl.db           247,308 GW rows │ 6-table star schema │ 53.8 MB
  │
  ├── ml/features.py      builds 4 position feature matrices; caches to .parquet
  │
  ├── ml/train.py         trains 21 models per position; serialises to models/
  │
  ├── ml/predict.py       inference on new GW feature rows; writes predictions CSV
  │
  └── run_gw.py (step 4)  compares predictions vs actuals; appends to monitoring_log.csv
```

**Star schema:**

| Table | Grain | Rows |
|---|---|---|
| `dim_season` | season | 10 |
| `dim_player` | player (stable cross-season `player_code`) | ~7,000 |
| `dim_team` | team × season | 200 |
| `dim_player_season` | player × season | ~40,000 |
| `fact_gw_player` | player × fixture | 247,308 |
| `fact_player_season_history` | player × prior season | ~30,000 |

---

## Modelling

**Target:** `total_points` — FPL points scored by a player in a single gameweek fixture.

**Training scope:** xG era (seasons 7–10, 2022-23 to 2025-26). Pre-xG seasons are excluded
due to missing xG/xA/xGI/xGC features and a −26.1% pts/GW structural drift. Post-filter
dataset: ~40,900 rows (GK 2,731 | DEF 13,723 | MID 19,495 | FWD 4,951).

**CV strategy:** Expanding-window temporal CV. Three folds:
- Fold 1: train 2022-23, validate 2023-24
- Fold 2: train 2022-24, validate 2024-25
- Fold 3: train 2022-25, validate 2025-26

CV folds extend automatically when a new xG-era season is added to `etl/schema.py`.

**21 models per position:**

| Tier | Models |
|---|---|
| 1 — Primary | Ridge, LightGBM |
| 2 — Extended | XGBoost, Random Forest, Extra Trees, Hist Gradient Boosting, MLP, BayesianRidge, ElasticNet, Lasso, Poisson GLM, Polynomial Ridge, Component, Minutes, FDR Mean, Last-Season Average, Position Mean, Baseline |
| 3 — Meta | Simple Average, Stacking, Blending |

**Ridge results (production model, post-Phase-8 alpha tuning):**

| Position | Best alpha | Mean MAE | Mean Spearman ρ |
|---|:---:|:---:|:---:|
| GK | 10.0 | 2.130 | 0.118 |
| DEF | 0.1 | 2.138 | 0.277 |
| MID | 0.1 | 1.830 | 0.371 |
| FWD | 0.1 | 2.270 | 0.413 |

**Best ensemble:** Blending (ridge + bayesian_ridge + poisson_glm + mlp) outperforms
Ridge solo on GK (MAE 2.123 vs 2.132) and DEF (2.121 vs 2.138).

**Monitoring alert thresholds (1.5× baseline MAE):** GK 3.494 | DEF 3.498 | MID 2.996 | FWD 3.609

---

## Dashboard

```bash
streamlit run outputs/dashboards/app.py
```

Opens at `http://localhost:8501`. The landing page shows the latest monitoring metrics and
the current top-5 predicted players per position.

| Page | What you can explore |
|---|---|
| Landing | Per-position MAE cards + top-10 predictions for the latest GW |
| 1 Data Explorer | Points distributions, home/away effect, team heatmap, player trajectories, xG scatter |
| 2 Bias & Quality | 10 quantified ML biases with mitigations, feature availability by era, known data quirks |
| 3 Model Performance | CV metrics table, calibration plots, MAE-by-fold, SHAP importances, residuals, learning curves |
| 4 GW Predictions | FDR fixture calendar, ranked predictions with uncertainty bands, CSV download |
| 5 Player Scouting | Boom/bust quadrant, value picks, form vs price, player comparison, price trajectory |
| 6 Database Explorer | 20 SQL templates, live table browser, free-form SQL editor, schema reference |

---

## Project Structure

```
fpl_analysis/
│
├── explore_dataset.py                     # Dataset scanner: column names + samples
├── LICENSE
├── README.md
├── REPORT.md                              # End-to-end portfolio report (all 9 phases)
├── requirements.txt
├── retrain_season.py                      # End-of-season retraining orchestrator (9 steps)
├── run_gw.py                              # Live post-GW pipeline: fetch → ETL → predict → monitor
│
├── data/                                  # Raw FPL CSVs — not in git (download from release)
│   ├── 2016-17/
│   │   ├── gws/
│   │   │   ├── merged_gw.csv             # PRIMARY fact source — one row per player per GW fixture
│   │   │   ├── gw1.csv ... gw38.csv      # Individual GW files (not used by pipeline)
│   │   │   └── xP1.csv ... xP38.csv      # Expected points files (not used by pipeline)
│   │   ├── players/
│   │   │   └── {Player_Name}/
│   │   │       ├── gw.csv                # Redundant with merged_gw — not used
│   │   │       └── history.csv           # AUTHORITATIVE for start_cost / end_cost
│   │   ├── cleaned_players.csv           # Subset of players_raw — not used
│   │   ├── player_idlist.csv             # Derivable from players_raw — not used
│   │   └── players_raw.csv               # Player + season dimension data (latest API snapshot)
│   ├── 2017-18/                          # Same structure for each of the 10 seasons
│   ├── 2018-19/
│   ├── 2019-20/                          # COVID season: GWs 1–29, then 39–47 (gap 30–38)
│   ├── 2020-21/
│   ├── 2021-22/
│   ├── 2022-23/                          # xG era begins (season_id = 7)
│   ├── 2023-24/
│   ├── 2024-25/                          # Manager era: mng_* columns
│   ├── 2025-26/                          # Defensive era: defensive_contribution, recoveries
│   ├── DATA_DICTIONARY.md
│   ├── master_team_list.csv              # Team ID → name map (2016-17 to 2023-24 only)
│   └── README.md
│
├── db/
│   └── fpl.db                            # SQLite database — rebuilt by python -m etl.run
│                                         # 247,308 GW rows | 6-table star schema | 53.8 MB
│
├── docs/
│   ├── code_review.md                    # Production-level code review (8.5 / 10)
│   ├── dashboard_report.md               # Phase 7: page specs, embedded visuals, integration checks
│   ├── data_biases.md                    # 10 quantified ML biases with mitigations
│   ├── deployment_report.md              # Phase 8: Ridge improvements, live pipeline, monitoring
│   ├── eda_report.md                     # EDA findings and Phase 4 recommendations
│   ├── feature_engineering.md            # Phase 4 report: features, outputs, Phase 5 implications
│   ├── modelling_evaluation_report.md    # Phase 5/6: CV results, stratifications, diagnostics
│   ├── modelling_plan.md                 # Model inventory, tier rationale, registry architecture
│   ├── monitoring_report.md              # Phase 9: schema alerting, eval reports, retraining
│   ├── phase7_plan.md                    # Phase 7 full implementation plan
│   ├── project_plan.md                   # Full pipeline specification — source of truth
│   └── schema_design.md                  # Star schema: all tables, columns, PKs, FKs, derivations
│
├── eda/
│   └── eda_report.ipynb                  # EDA notebook (25 code cells, 30 markdown cells)
│
├── etl/
│   ├── __init__.py
│   ├── fetch.py                          # FPL API client with exponential-backoff retry logic
│   ├── loaders.py                        # One loader function per table (6 functions)
│   ├── run.py                            # Entry point: python -m etl.run
│   ├── schema.py                         # DDL, SEASONS tuple, era flags, EXPECTED_COLS per season
│   └── validate.py                       # 10 post-load assertion checks (raises on failure)
│
├── logs/
│   ├── db_check_casestudy.md             # Case-study cross-table validation log
│   ├── db_check_full.md                  # Full logical soundness audit (32 checks)
│   ├── monitoring/
│   │   ├── gw{N}_s{season}_eval.md       # Per-GW narrative evaluation reports
│   │   ├── monitoring_log.csv            # Per-GW MAE / RMSE / Spearman / top-10 prec / alert
│   │   └── schema_alerts.csv             # Schema change alerts: season_id, gw, check_type, columns
│   └── training/
│       ├── cv_metrics_all.csv            # Combined CV metrics across all positions
│       ├── cv_metrics_{GK|DEF|MID|FWD}.csv
│       ├── cv_metrics_{GK|DEF|MID|FWD}_seq.csv   # LSTM/GRU sequential model CV results
│       ├── cv_preds_{GK|DEF|MID|FWD}.parquet     # OOF predictions used for meta-model training
│       ├── cv_report_{GK|DEF|MID|FWD}.md         # Per-position CV summary report
│       └── ridge_alpha_search.csv        # Alpha grid search results per position and fold
│
├── ml/
│   ├── evaluate.py                       # Expanding-window CV, metrics, SHAP, calibration plots
│   ├── evaluate_phase6.py                # Post-hoc analysis: minutes bucket, price band, residuals
│   ├── evaluate_sequential.py            # Standalone LSTM/GRU CV pipeline (PyTorch)
│   ├── features.py                       # build_feature_matrix() — 20+ features, parquet cache
│   ├── models.py                         # Central registry: ModelSpec dataclass, 21 models
│   ├── predict.py                        # Inference on new GW data: python -m ml.predict
│   └── train.py                          # Training + serialisation (--all / --meta / --alpha-search)
│
├── models/                               # 168 serialised artefacts — not in git
│   ├── DEF_baseline.pkl                  # Naming: {GK|DEF|MID|FWD}_{model_name}.pkl
│   ├── DEF_baseline_meta.json            # Metadata: CV MAE, feature list, alpha, timestamp
│   ├── ... (21 models × 4 positions × 2 files = 168 total)
│   └── v{season}/                        # Archived models from previous season (retrain_season.py)
│
└── outputs/
    ├── dashboards/
    │   ├── app.py                        # Landing page: monitoring summary + top predictions
    │   ├── utils.py                      # Shared loaders: query_db, load_predictions, load_oof, etc.
    │   ├── .streamlit/
    │   │   └── config.toml               # Wide layout, light theme, headless config
    │   └── pages/
    │       ├── 1_Data_Explorer.py        # Points distributions, home/away, team heatmap, xG scatter
    │       ├── 2_Bias_Quality.py         # 10 ML biases, era comparison, fixture difficulty
    │       ├── 3_Model_Performance.py    # CV table, calibration, MAE-by-fold, SHAP, residuals
    │       ├── 4_GW_Predictions.py       # FDR calendar, ranked predictions, uncertainty bands
    │       ├── 5_Player_Scouting.py      # Boom/bust quadrant, value picks, career stats
    │       └── 6_Database_Explorer.py    # 20 SQL templates + free-form SQL editor
    ├── eda/                              # 22 exported EDA charts (PNG)
    │   ├── era_comparison.png
    │   ├── home_away_effect.png
    │   ├── points_distribution.png
    │   ├── team_strength_heatmap.png
    │   ├── top6_fixture_effect.png
    │   └── ... (22 charts total)
    ├── features/                         # Cached feature matrices — clear before retraining
    │   ├── feature_matrix_DEF.parquet
    │   ├── feature_matrix_FWD.parquet
    │   ├── feature_matrix_GK.parquet
    │   └── feature_matrix_MID.parquet
    ├── models/                           # Diagnostic plots (one set per position)
    │   ├── calibration_{GK|DEF|MID|FWD}.png
    │   ├── learning_curves.png
    │   ├── mae_by_fold_{GK|DEF|MID|FWD}.png
    │   ├── residuals_{GK|DEF|MID|FWD}.png
    │   └── shap_{GK|DEF|MID|FWD}.png
    └── predictions/                      # Per-GW prediction CSVs (written by run_gw.py)
        ├── gw24_s10_predictions.csv
        └── gw30_s10_predictions.csv
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/syed-taha-ali/premier-league-player-analytics.git
cd premier-league-player-analytics
pip install -r requirements.txt
```

PyTorch is required for the LSTM/GRU evaluation module. If you only want the tabular
pipeline, it can be omitted — all other modules function normally without it.

### 2. Download the dataset

The raw CSV data is not stored in git. Download it from the GitHub release:

```bash
gh release download v1.0-data --pattern "data.zip"
unzip data.zip && rm data.zip
```

Or download `data.zip` manually from the
[v1.0-data release](https://github.com/syed-taha-ali/premier-league-player-analytics/releases/tag/v1.0-data)
and unzip into the repo root.

### 3. Build the database

```bash
python -m etl.run
```

Drops and rebuilds `db/fpl.db` from the CSV files (~16s). Prints PASS/FAIL for all 10
validation checks on completion.

### 4. Train models

```bash
# Cross-validation evaluation (writes metrics and OOF parquets to logs/training/)
python -m ml.evaluate

# Train all tabular and decomposed models
python -m ml.train --all

# Train meta-models (requires OOF predictions from the evaluate step)
python -m ml.train --meta
```

Pre-trained artefacts are included in `models/` — skip this step to use them directly.

### 5. Generate predictions

```bash
# Predict the latest available GW using the default models
python -m ml.predict

# Predict a specific GW
python -m ml.predict --gw 31 --season 10 --models ridge blending --top 10
```

### 6. Run the live post-GW pipeline

```bash
# Full pipeline: fetch API data, rebuild DB, predict, log metrics
python run_gw.py --gw 31 --season 10

# Skip API fetch (use existing CSVs)
python run_gw.py --gw 31 --season 10 --skip-fetch

# Skip ETL rebuild (use existing fpl.db)
python run_gw.py --gw 31 --season 10 --skip-etl
```

**Important:** clear `outputs/features/*.parquet` after adding new GWs — the feature
matrix cache becomes stale and predictions will be empty until it is rebuilt.

### 7. Launch the dashboard

```bash
streamlit run outputs/dashboards/app.py
```

Opens at `http://localhost:8501`.

### 8. Preview end-of-season retraining

```bash
python retrain_season.py --season 11 --dry-run
```

---

## Data

Raw FPL data sourced from the
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
dataset for historical seasons. Current-season data is fetched live via `etl/fetch.py`.

Each season folder contains three authoritative source files:

| File | Role |
|---|---|
| `{season}/gws/merged_gw.csv` | Primary fact source — one row per player per GW fixture |
| `{season}/players_raw.csv` | Player + season dimension data (latest API snapshot) |
| `{season}/players/{name}/history.csv` | Authoritative source for start_cost / end_cost |

To fetch the current season's latest GW manually:

```bash
python -m etl.fetch --gw 31 --season 2025-26
```

Please cite the source dataset if you use this data in your own work:

```
@misc{anand2016fantasypremierleague,
  title        = {{FPL Historical Dataset}},
  author       = {Anand, Vaastav},
  year         = {2022},
  howpublished = {Retrieved August 2022 from \url{https://github.com/vaastav/Fantasy-Premier-League/}}
}
```

---

## Results and Insights

**1. Fixture context is the dominant scoring signal across all positions**
Home advantage produces a significant scoring premium: DEF +18.7%, MID +11.1%,
FWD +11.2%, GK +7.5% pts/GW on average. Facing a top-6 opponent imposes a symmetric
penalty: DEF −33.8%, FWD −21.2%, GK −17.6%, MID −16.6%.

**2. Team defensive quality explains the majority of GK and DEF scoring variance**
Cumulative goals conceded by a player's own team (`team_goals_conceded_season`) accounts
for 46.6% of scoring variance for GK and DEF — more than any individual player metric.
For outfield attackers, personal form dominates; for backline players, team context is
the primary signal.

**3. Ridge outperforms LightGBM across all four positions without hyperparameter tuning**
The linear model's L2 regularisation generalises better than LightGBM's default tree
configuration on datasets of 5,000–20,000 training rows per fold. LightGBM's advantage
is expected to increase once Optuna tuning is applied, particularly for FWD where
non-linear interactions between xG rate and opponent rank are strongest.

**4. The blending ensemble delivers the best results for defensive positions**
Blending (ridge + bayesian_ridge + poisson_glm + mlp) achieves MAE 2.123 for GK and
2.121 for DEF — outperforming Ridge alone (2.130 and 2.138). The BayesianRidge component
also provides per-player prediction uncertainty, surfaced as confidence bands in the GW
Predictions dashboard page.

---

## Limitations

- **xG era only for modelling.** Pre-2022-23 data lacks xG/xA/xGI/xGC features and shows
  a −26.1% scoring drift, making it incompatible with the current feature set. The
  training corpus is four seasons (~40,900 rows).
- **Unofficial FPL API.** The schema has changed between seasons (new columns added in
  2024-25, columns dropped in 2025-26). The schema alerting layer catches these changes
  but may require a manual `EXPECTED_COLS` update each season.
- **LSTM/GRU not integrated into the live pipeline.** Sequential models are evaluated in
  `ml/evaluate_sequential.py` and are not serialised for use in `run_gw.py`.
- **Prediction intervals from BayesianRidge only.** Tree-based models produce point
  predictions without native uncertainty estimates.
- **Double Gameweeks.** For a player's second DGW fixture, rolling features already
  include the first fixture of that same GW — a minor within-GW lookahead that cannot
  be resolved without fixture-scheduling data at prediction time.
- **LightGBM not yet Optuna-tuned.** Current results use heuristic baseline
  hyperparameters. FWD is the highest priority for tuning.

---

## Future Improvements

- **Optuna LightGBM tuning** — `python -m ml.train --tune --position FWD` to close the
  remaining gap between Ridge and LightGBM on FWD
- **Player name resolution** — join `dim_player` on `player_code` in prediction outputs
  and the dashboard prediction table (currently shows integer `player_code`)
- **Automated threshold recalibration** — compute updated alert thresholds (1.5× new
  baseline MAE) automatically during `retrain_season.py` step 9
- **Transfer optimisation layer** — solve a constrained team selection problem using
  predictions (budget, squad rules, captaincy multiplier, transfer cost)
- **Season 11 readiness** — adding `has_xg_stats=1` to `SEASONS` in `etl/schema.py`
  and making `season_id` detection dynamic in `run_gw.py` are the two required code
  changes when season 11 data becomes available
