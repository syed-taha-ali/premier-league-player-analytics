# FPL Analysis — Fantasy Premier League Prediction Pipeline

An end-to-end data science pipeline for predicting Fantasy Premier League player points,
built across 9 phases: data collection, ETL, EDA, feature engineering, modelling,
evaluation, an interactive dashboard, a live GW pipeline, and a monitoring layer.

---

## What this does

- Ingests 10 seasons of FPL data (2016-17 to 2025-26) into a SQLite star schema
- Trains position-specific Ridge and LightGBM models using an expanding-window CV strategy
- Predicts GW-level points per player after each gameweek
- Surfaces everything in a 6-page Streamlit dashboard: historical EDA, model diagnostics,
  ranked predictions, player scouting tools, and a 20-template SQL explorer

**Production model CV MAE (Ridge):** GK 2.130 | DEF 2.138 | MID 1.835 | FWD 2.270

---

## Quick start

### 1. Clone the repo

```bash
git clone https://github.com/syed-taha-ali/premier-league-player-analytics.git
cd premier-league-player-analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

The raw data is not stored in git. Download it from the GitHub release:

```bash
gh release download v1.0-data --pattern "data.zip"
unzip data.zip
rm data.zip
```

Or download `data.zip` manually from the
[v1.0-data release](https://github.com/syed-taha-ali/premier-league-player-analytics/releases/tag/v1.0-data)
and unzip it into the repo root.

### 4. Build the database

```bash
python -m etl.run
```

Drops and rebuilds `db/fpl.db` from scratch (~15s). Runs 10 validation checks on
completion.

### 5. Launch the dashboard

```bash
streamlit run outputs/dashboards/app.py
```

Opens at `http://localhost:8501`.

---

## Live GW pipeline

After each gameweek result is published, run:

```bash
python run_gw.py --gw <N> --season 10
```

This fetches live data from the FPL API, rebuilds the database, generates predictions,
and appends performance metrics to the monitoring log. Useful flags:

| Flag | Effect |
|------|--------|
| `--skip-fetch` | Skip API fetch, use existing CSVs |
| `--skip-etl` | Skip ETL, use existing `db/fpl.db` |
| `--model ridge lgbm` | Override default model set |

**Note:** clear `outputs/features/*.parquet` after adding new GWs — the feature matrix
cache becomes stale and predictions will be empty.

---

## Project structure

```
fpl_analysis/
├── data/                   # Raw FPL CSVs (download from release — not in git)
├── db/                     # SQLite database — rebuilt by python -m etl.run
├── etl/                    # ETL pipeline: schema, loaders, validation, API fetch
├── ml/                     # Feature engineering, model registry, training, evaluation, inference
├── models/                 # 168 serialised model artefacts (.pkl + _meta.json)
├── outputs/
│   ├── dashboards/         # Streamlit app (app.py + 6 pages + utils.py)
│   ├── eda/                # Exported EDA charts (PNG)
│   ├── features/           # Cached feature matrices (parquet — auto-rebuilt)
│   ├── models/             # Diagnostic plots (calibration, SHAP, residuals, learning curves)
│   └── predictions/        # Per-GW prediction CSVs
├── logs/
│   ├── training/           # CV metrics, OOF parquets, alpha search results
│   └── monitoring/         # Per-GW MAE log, schema alerts, narrative eval reports
├── docs/                   # Phase reports, bias analysis, EDA findings, modelling plan
├── run_gw.py               # End-to-end GW runner
├── retrain_season.py       # End-of-season retraining orchestrator
└── project_plan.md         # Full pipeline specification — source of truth
```

---

## Dashboard pages

| Page | Contents |
|------|----------|
| Landing | Per-position MAE metric cards + top-10 predictions for the latest GW |
| 1 — Data Explorer | Points distributions, home/away effect, team strength heatmap, player career trajectories, xG scatter, era comparison |
| 2 — Bias & Data Quality | ML bias analysis, feature availability by era, fixture difficulty effect, price vs performance, known data quirks |
| 3 — Model Performance | CV comparison table, OOF calibration scatter, MAE-by-fold, SHAP, residuals, learning curves, monitoring trend, per-GW eval report viewer |
| 4 — GW Predictions | FDR calendar, captain cards, filterable prediction table with uncertainty, ownership bubble chart, CSV download |
| 5 — Player Scouting | Boom/bust quadrant, value picks, form vs price, player comparison, price trajectory, component model analysis |
| 6 — Database Explorer | 20 SQL templates, table browser, free-form SQL editor, schema reference |

---

## Key design decisions

- **Era scope:** xG era only (seasons 7–10, 2022-23 to 2025-26). Pre-xG seasons excluded
  due to missing xG/xA/xGI/xGC features and a −26.1% pts/GW scoring drift.
- **Production model:** Ridge for all positions. Beats LightGBM on every metric at default
  hyperparameters.
- **CV strategy:** Expanding-window temporal CV (season 7→8, 7-8→9, 7-9→10). Never random
  split.
- **Mandatory features:** `was_home`, `opponent_season_rank`, `team_goals_conceded_season`.
- **Leakage policy:** `bonus`, `bps`, `ict_index`, same-GW `clean_sheets`/`goals_scored`/
  `assists` are banned as features. Transfer signals lagged by 1 GW.

Full rationale in `project_plan.md` and `docs/`.

---

## Data source

Raw FPL data sourced from the
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
dataset. Please cite the original work if you use this data:

```
@misc{anand2016fantasypremierleague,
  title  = {{FPL Historical Dataset}},
  author = {Anand, Vaastav},
  year   = {2022},
  howpublished = {Retrieved August 2022 from \url{https://github.com/vaastav/Fantasy-Premier-League/}}
}
```
