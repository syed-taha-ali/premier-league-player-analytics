# Phase 7 — Interactive Dashboard & Visualisations: Implementation Plan

## Status: COMPLETE (2026-03-18)

All phases A–H implemented and verified. Dashboard serves at `streamlit run outputs/dashboards/app.py`.
All 21 integration checks pass. HTTP 200 confirmed on local launch.

## Context

Phases 1–6, 8, and 9 are complete. The pipeline produces serialised models, per-GW prediction
CSVs, CV metrics, monitoring logs, and a full suite of static charts. Phase 7 surfaces all of
this through an interactive local dashboard so that weekly GW decisions can be made in one place:
explore historical data, review model performance, view ranked predictions, and identify value picks.

---

## Framework Decision: Streamlit

Streamlit is selected over Plotly Dash.

Rationale:
- Simpler syntax for data tables, filters, and downloads — matches the project's local/analytical
  use case better than a production web framework
- `@st.cache_data` caching is trivially correct for CSV/parquet/SQL data loading
- Multi-page routing is built-in via the `pages/` directory convention
- No callback wiring needed; state is managed by widget return values

---

## File Layout

All dashboard files live under `outputs/dashboards/`:

```
outputs/dashboards/
├── .streamlit/
│   └── config.toml            # Wide layout, page title, usage stats off
├── app.py                     # Landing page — project overview and navigation guide
├── utils.py                   # Shared: PROJECT_ROOT, DB queries, data loaders, caching
└── pages/
    ├── 1_Data_Explorer.py     # Page 1: season/position EDA
    ├── 2_Bias_Quality.py      # Page 2: data quality and known biases
    ├── 3_Model_Performance.py # Page 3: CV metrics, calibration, SHAP
    ├── 4_GW_Predictions.py    # Page 4: ranked GW prediction table
    ├── 5_Player_Scouting.py   # Page 5: value picks and rotation risk
    └── 6_Database_Explorer.py # Page 6: preset queries, table browser, free-form SQL
```

Run from the project root:
```bash
streamlit run outputs/dashboards/app.py
```

### `.streamlit/config.toml`

Must be created at `outputs/dashboards/.streamlit/config.toml`:

```toml
[server]
headless = true

[browser]
gatherUsageStats = false

[theme]
base = "light"

[client]
showErrorDetails = true

[global]
# layout="wide" applied per-page via st.set_page_config — see app.py and each page file
```

Each page file calls `st.set_page_config(layout="wide", page_title="FPL Analysis — <Page Name>")`
as its first statement. Without `layout="wide"`, all data tables and charts are rendered in a
narrow centred column — unusable for wide DataFrames.

---

## Dependencies

Add to a new `requirements.txt` at the project root (or append to existing):
```
streamlit>=1.35
plotly>=5.20
```

All other imports (`pandas`, `numpy`, `sqlite3`, `joblib`, `pathlib`) are already present in
the project environment.

---

## Shared Utilities (`utils.py`)

`utils.py` provides the shared data-loading layer for all pages. All loaders use
`@st.cache_data` to prevent redundant I/O on page re-runs.

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # fpl_analysis/
DB_PATH      = PROJECT_ROOT / "db" / "fpl.db"
PRED_DIR     = PROJECT_ROOT / "outputs" / "predictions"
EDA_DIR      = PROJECT_ROOT / "outputs" / "eda"
MODELS_DIR   = PROJECT_ROOT / "outputs" / "models"
LOG_DIR      = PROJECT_ROOT / "logs" / "monitoring"
TRAIN_DIR    = PROJECT_ROOT / "logs" / "training"
```

Functions to implement in `utils.py`:

| Function | Returns | Notes |
|----------|---------|-------|
| `load_player_names()` | `{player_code: web_name}` | `SELECT player_code, web_name FROM dim_player` |
| `load_team_names()` | `{team_sk: team_name}` | `SELECT team_sk, team_name FROM dim_team` |
| `load_predictions(gw, season_id)` | DataFrame | Load CSV; join player + team names from DB |
| `list_available_gws()` | list of `(gw, season_id)` tuples | Glob `outputs/predictions/gw*_s*_predictions.csv`; parse names |
| `load_monitoring_log()` | DataFrame | Read `logs/monitoring/monitoring_log.csv` |
| `load_cv_metrics()` | DataFrame | Read `logs/training/cv_metrics_all.csv` |
| `query_db(sql)` | DataFrame | `pd.read_sql(sql, sqlite3.connect(DB_PATH))` |

**Naming convention for `list_available_gws()`:**
Filenames are `gw{N}_s{season}_predictions.csv`. Parse with:
```python
re.match(r"gw(\d+)_s(\d+)_predictions\.csv", fname)
```
Sort descending by `(season_id, gw)` so the latest GW is the default.

**Opponent display:**
The prediction CSV does not carry opponent team name. Resolve via:
```sql
SELECT DISTINCT
    f.season_id, f.gw, f.fixture_id, f.player_code,
    dt.team_name AS opponent_team
FROM fact_gw_player f
JOIN dim_team dt ON dt.team_sk = f.opponent_team_sk
WHERE f.season_id = {season_id} AND f.gw = {gw}
```
Join this onto the prediction DataFrame on `(season_id, gw, fixture_id, player_code)` in
`load_predictions()`.

**DB connection — read-only:**
All DB access uses a read-only URI to prevent accidental writes while ETL may be running:
```python
def query_db(sql: str, params=()) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
```
Use parameterised queries (`?` placeholders + `params` tuple) throughout to prevent SQL injection
from free-form inputs in the DB Explorer.

**Error and empty-state handling:**
Every page must handle the following gracefully without raising an unhandled exception:

| Condition | Handling |
|-----------|----------|
| `outputs/predictions/` empty (no CSVs yet) | `st.warning("No prediction files found. Run run_gw.py to generate predictions.")` |
| `db/fpl.db` missing or locked (ETL running) | Catch `sqlite3.OperationalError`; `st.error("Database unavailable — ETL may be running. Try again in a moment.")` |
| `logs/monitoring/monitoring_log.csv` empty | `st.info("No monitoring data yet. Run run_gw.py after the first GW.")` |
| Feature matrix parquet missing or stale | `st.warning("Feature matrix cache missing. Run build_feature_matrix to regenerate.")` |
| Selected GW has no actuals yet (`total_points` all NaN) | Show "Actual pts" column as "—"; add caption "Actuals will populate after GW results are published." |

**pred_std gap:**
`bayesian_ridge` predictions in the current CSVs do not include `pred_std`. Add this as a one-line
change to `ml/predict.py`: when the `bayesian_ridge` model is in the active models list, call
`model.predict(X, return_std=True)` and write `pred_bayesian_ridge_std` to the output CSV. The
predictions page will then expose this column as the "uncertainty band" when bayesian_ridge is
the selected display model.

---

## Page 1 — Data Explorer (`1_Data_Explorer.py`)

**Purpose:** Interactive EDA — historical scoring distributions, home/away splits, team strength, and player career trajectories.

### Controls (sidebar)

| Widget | Type | Default |
|--------|------|---------|
| Season(s) | st.multiselect | All seasons (7–10) |
| Position | st.selectbox | "All" |

### Section A — Points Distribution

Interactive Plotly histogram from a DB query:
```sql
SELECT season_id, position_label AS position, total_points
FROM fact_gw_player fgp
JOIN dim_player_season dps USING (season_id, player_code)
WHERE total_points > 0 AND season_id IN (...)
```
Chart: `px.histogram(color="position", nbins=40, facet_col="season_id")`.
This replaces the static `outputs/eda/points_distribution.png` with an interactive equivalent.

### Section B — Home vs Away Effect

Bar chart of mean points by position × `was_home`:
```sql
SELECT position_label AS position, was_home, AVG(total_points) AS mean_pts
FROM fact_gw_player fgp
JOIN dim_player_season dps USING (season_id, player_code)
WHERE season_id IN (...) AND total_points > 0
GROUP BY position_label, was_home
```
Chart: `px.bar(barmode="group", x="position", y="mean_pts", color="was_home")`.

### Section C — Team Strength Heatmap

Mean goals conceded per team per season (use `team_h_score`/`team_a_score` as per CLAUDE.md —
never the player-level `goals_conceded` column).

```sql
SELECT
    s.season_name,
    dt.team_name,
    AVG(f.team_goals_conceded_season) AS avg_goals_conceded
FROM fact_gw_player f
JOIN dim_team dt ON dt.team_sk = f.team_sk
JOIN dim_season s ON s.season_id = f.season_id
WHERE f.season_id IN (...)
GROUP BY s.season_name, dt.team_name
```
Chart: `px.imshow(pivot_table)` heatmap with team on y-axis, season on x-axis.

### Section D — Player Career Trajectory

`st.text_input` for player name search → query `dim_player` for partial `web_name` match →
`st.selectbox` to pick from matches → line chart of `AVG(total_points)` per (season_id, gw).

```sql
SELECT gw, season_id, total_points
FROM fact_gw_player
WHERE player_code = {player_code}
ORDER BY season_id, gw
```
Chart: `px.line(x="gw", y="total_points", color="season_id")` with a "points per GW" annotation.

### Section E — xG vs Actual Goals (xG Era)

Interactive scatter validating the xG data: does expected goals predict actual goals?

```sql
SELECT
    dp.web_name,
    dps.position_label          AS position,
    ds.season_name              AS season,
    SUM(f.expected_goals)       AS xg,
    SUM(f.goals_scored)         AS goals,
    COUNT(DISTINCT f.fixture_id) AS appearances
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code AND dps.season_id = f.season_id
JOIN dim_season        ds  ON ds.season_id    = f.season_id
WHERE f.season_id >= 7 AND f.minutes > 0
GROUP BY f.player_code, f.season_id
HAVING appearances >= 5
```

Chart: `px.scatter(x="xg", y="goals", color="position", hover_data=["web_name", "season", "appearances"])`.
Add an `x=y` reference line (`px.add_shape`) — points above the line = overperforming xG,
below = underperforming. Filter by position and season via sidebar controls.

### Section F — Era Comparison

Embed `outputs/eda/era_comparison.png` (pts/GW by era — already generated). Add a caption
noting the −26.1% pts/GW drift in pre-xG seasons that motivated the xG-era-only ML scope.

### Section G — Team Attack vs Defence Strength

Page 1 Section C currently shows only defensive strength (goals conceded). For full fixture
planning you need both axes: how dangerous is this team offensively (risk to opposing GK/DEF)
and how leaky defensively (opportunity for opposing FWD/MID).

```sql
SELECT
    dt.team_name,
    ds.season_name,
    -- Offensive: goals this team scored across all fixtures (aggregate from opponent perspective)
    AVG(f.team_goals_scored_rolling_3gw)   AS attack_strength,
    -- Defensive: goals this team conceded
    AVG(f.team_goals_conceded_season)       AS defensive_weakness,
    COUNT(DISTINCT f.fixture_id)            AS fixtures
FROM fact_gw_player f
JOIN dim_team   dt ON dt.team_sk  = f.team_sk
JOIN dim_season ds ON ds.season_id = f.season_id
WHERE f.season_id IN ({selected_seasons})
  AND f.minutes > 0
GROUP BY dt.team_name, ds.season_name
```

Chart: `px.scatter(x="defensive_weakness", y="attack_strength", color="season_name",
hover_data=["team_name"], text="team_name")`. Teams in the top-right quadrant (high attack,
high defensive weakness) are the most volatile fixture for all positions — great for attackers,
risky for GK/DEF. Teams in the bottom-left are the safest fixtures for GK/DEF. Add quadrant
reference lines at the median of each axis.

---

## Page 2 — Bias & Data Quality (`2_Bias_Quality.py`)

**Purpose:** Surface known ML biases and data quality characteristics.

### Section A — Bias Summary Table

Parse `docs/data_biases.md` at load time and render as `st.dataframe`. The file contains 10
quantified biases with mitigation notes. Extract the structured sections as a DataFrame with
columns: Bias Name, Magnitude, Direction, Mitigation.

Alternatively (simpler), render the markdown directly: `st.markdown(Path(docs/.../data_biases.md).read_text())`.

### Section B — Feature Availability by Era

Embed `outputs/eda/missing_data_matrix.png` via `st.image()`. Add a caption explaining the
xG era column restriction (seasons 7–10 only).

### Section C — Fixture Difficulty Effect

Embed `outputs/eda/top6_fixture_effect.png` with caption. Add an interactive companion:
scatter plot of `opponent_season_rank` vs `total_points` from the feature matrix parquet
(loaded with `pd.read_parquet(FEATURES_DIR / "feature_matrix_{pos}.parquet")`).

### Section D — Price vs Performance

Embed `outputs/eda/price_vs_season_points.png`. Add `outputs/eda/price_band_performance.png`
alongside it.

### Section E — Known Data Quirks

A dedicated section surfacing the 7 documented data quirks from CLAUDE.md that affect
analysis and interpretation. Render as a styled table — users need to know these when
interpreting any chart or query result:

| Quirk | Seasons affected | Impact |
|-------|-----------------|--------|
| 2019-20 COVID gap | Season 4 | GWs 30–38 never played; season runs GW1–29 then GW39–47. `total_gws=47`. Rolling windows must not cross the gap. |
| 2022-23 GW7 absent | Season 7 | Postponed for national mourning; absorbed into other GWs. Season has 37 data GWs. |
| Ferguson points discrepancy | Season 9 (2024-25) | 1-pt and 17-min divergence between dim_player_season and fact_gw_player sums. Source data artefact. |
| Manager rows (2024-25) | Season 9 | FPL manager game mode adds rows with `value` as low as 5 and `mng_*` columns populated. Filtered out by `mng_win IS NULL` in ML pipeline. |
| `starts` present in 2025-26 | Season 10 | Schema flags `has_starts=0` for season 10, but data has `starts` populated across all 18,173 rows. |
| `goals_conceded` is time-on-pitch scoped | All seasons | Reflects goals conceded while that player was on the pitch, not the team's match total. 75.9% of team-fixtures show inconsistent values across players. Never use for team-level defensive stats. |
| `master_team_list.csv` missing 2024-25 and 2025-26 | Seasons 9–10 | Team mapping for recent seasons derived by cross-joining `players_raw.team` with `merged_gw.team`. Not a DB concern but relevant for raw data access. |

Render with `st.table()` or `st.dataframe()`. Add a `st.warning` banner at the top of the
section: "These quirks are handled in the ETL and ML pipeline. Be aware of them when writing
custom SQL queries or interpreting raw data."

---

## Page 3 — Model Performance (`3_Model_Performance.py`)

**Purpose:** CV results, calibration, SHAP, and live monitoring trend.

### Section A — CV Model Comparison Table

Load `logs/training/cv_metrics_all.csv` (schema: `fold, model, mae, rmse, r2, spearman, top10_prec, position`).

Compute mean across folds:
```python
summary = df.groupby(["position", "model"])[["mae", "rmse", "spearman", "top10_prec"]].mean().round(3)
```

Display as `st.dataframe` with position filter (st.selectbox). Highlight the ridge row in
each position block using `st.dataframe(summary.style.highlight_min(subset=["mae"]))`.

### Section B — Interactive OOF Calibration

The `logs/training/cv_preds_{pos}.parquet` files contain out-of-fold predictions for all
21 tabular models with player_code, actual `total_points`, and `fold` indicator. Use these
to build a hover-enabled interactive scatter that is far more informative than the static PNG.

Controls (in-page, not sidebar):
- `st.selectbox("Model")` — any of the 21 model columns in the parquet
- `st.selectbox("Fold")` — 1, 2, 3, or "All"
- Position selector already set by the page-level tabs (see Section D below)

```python
@st.cache_data
def load_oof(position: str) -> pd.DataFrame:
    path = TRAIN_DIR / f"cv_preds_{position}.parquet"
    df   = pd.read_parquet(path)
    names = utils.load_player_names()
    df["player"] = df["player_code"].map(names)
    return df
```

Chart:
```python
fig = px.scatter(
    df_filtered,
    x=f"pred_{model}",
    y="total_points",
    color="fold",
    hover_data=["player", "season_id", "gw", "total_points"],
    labels={"x": f"Predicted pts ({model})", "y": "Actual pts"},
    opacity=0.5,
)
fig.add_shape(type="line", x0=0, y0=0, x1=20, y1=20, line=dict(dash="dash", color="grey"))
```

The `x=y` reference line separates over-predictions (above line) from under-predictions.
Hovering reveals the specific player and GW — answering "who is that 20-point outlier the
model only gave 3?" directly.

### Section C — Static Diagnostic Plots (by position)

Retain the static plots for SHAP, MAE-by-fold, and learning curves — these don't benefit from
interactivity. Display in a 2×2 grid using `st.columns`:

| Column 1 | Column 2 |
|----------|----------|
| `mae_by_fold_{pos}.png` | `shap_{pos}.png` |
| `learning_curves.png` | *(empty or residual decomposition — see Section E)* |

Use `st.image(path, caption=..., use_column_width=True)`.

### Section C — Learning Curves

Embed `outputs/models/learning_curves.png` (single file covering all 4 positions).

### Section D — Live Monitoring Trend

Load `logs/monitoring/monitoring_log.csv` (schema: `season_id, gw, model, position, mae, rmse, spearman, top10_precision, rolling_mae_5gw, threshold, alert, logged_at`).

Line chart of `rolling_mae_5gw` per position over GW, with a horizontal threshold line per
position (from the CSV `threshold` column). Color points red where `alert = "FAIL"`.

Chart: `px.line(x="gw", y="rolling_mae_5gw", color="position")` + `px.add_hline()` per position.

If the monitoring log has fewer than 2 GWs, show `st.info("Monitoring trend will populate as
more GWs are run. Currently {n} GW(s) logged.")`.

### Section E — Residual Decomposition by Feature Bucket

Shows *where* the model is systematically wrong — the most actionable model diagnostic in
the entire dashboard. Join `cv_preds_{pos}.parquet` with `fact_gw_player` on
`(season_id, gw, fixture_id, player_code)` to attach feature values to each OOF prediction.

Compute residual: `residual = total_points - pred_{model}` (positive = underestimated).

Then group by four feature buckets and show mean residual as bar charts in `st.columns(2)`:

**Bucket 1 — Home vs Away:**
```python
df["venue"] = df["was_home"].map({1: "Home", 0: "Away"})
group = df.groupby("venue")["residual"].mean()
# px.bar — if home residual is consistently positive, model underestimates home premium
```

**Bucket 2 — Opponent Rank Tier:**
```python
df["opp_tier"] = pd.cut(df["opponent_season_rank"],
                         bins=[0,6,12,16,18,20],
                         labels=["Top 6","Mid-upper","Mid-lower","Bottom 4","Bottom 2"])
```

**Bucket 3 — Price Band:**
```python
df["price_band"] = pd.cut(df["start_cost"] / 10,
                           bins=[0,5,7,9,11,20],
                           labels=["<£5m","£5-7m","£7-9m","£9-11m",">£11m"])
```

**Bucket 4 — Minutes Bucket:**
```python
df["mins_bucket"] = pd.cut(df["minutes"], bins=[0,45,60,75,90,200],
                            labels=["<45","45-60","60-75","75-90","90+"])
```

A bar in positive territory means the model underestimates points in that bucket (safe to
predict higher). Negative means overestimates (model too optimistic for that segment). This
directly informs where the model needs improvement and what to adjust for in FPL decisions.

### Section F — Per-GW Narrative Reports

The pipeline writes a full narrative eval report after each GW (`logs/monitoring/gw{N}_s{season}_eval.md`).
These contain position-level summaries, top predictions vs actuals, largest misses, and alert
status — rich context that is currently invisible in the dashboard.

Controls:
- `st.selectbox("Gameweek report")` — auto-populated by globbing `logs/monitoring/gw*_s*_eval.md`;
  sorted descending so the latest GW is the default

Implementation:
```python
report_files = sorted(LOG_DIR.glob("gw*_s*_eval.md"), reverse=True)
labels = [f.stem for f in report_files]   # e.g. "gw30_s10_eval"
selected = st.selectbox("Gameweek report", labels)
content  = (LOG_DIR / f"{selected}.md").read_text()
st.markdown(content)
```

If no report files exist yet, show `st.info("No GW reports yet. Run run_gw.py after the first GW.")`.

---

## Page 4 — GW Predictions (`4_GW_Predictions.py`)

**Purpose:** Primary weekly decision tool — ranked player predictions with filters and download.

### Controls (sidebar)

| Widget | Type | Default | Notes |
|--------|------|---------|-------|
| GW selector | st.selectbox | Latest available | Populated from `list_available_gws()` |
| Model | st.selectbox | ridge | Options: ridge, bayesian_ridge, blending, ensemble |
| Position | st.multiselect | All | GK, DEF, MID, FWD |
| Price band (£m) | st.slider | 4.0–15.0 | Filter by `value_lag1 / 10` |
| Top-N per position | st.number_input | 20 | Show top-N ranked by prediction |
| Show differentials only | st.checkbox | False | Filter to low-ownership high-predicted picks (see below) |

### Section A — Fixture Difficulty Calendar

The single most-used FPL planning visualization — completely absent from the current plan.
A team × GW heatmap showing fixture difficulty for every team across every played GW in the
selected season. FDR derived from opponent season rank (same 1–5 bucketing as the main table).

```sql
SELECT DISTINCT
    dt_home.team_name   AS team,
    f.gw,
    dt_opp.team_name    AS opponent,
    f.was_home,
    f.opponent_season_rank,
    -- Derive FDR bucket
    CASE
        WHEN f.opponent_season_rank <= 6  THEN 5
        WHEN f.opponent_season_rank <= 10 THEN 4
        WHEN f.opponent_season_rank <= 14 THEN 3
        WHEN f.opponent_season_rank <= 17 THEN 2
        ELSE 1
    END AS fdr
FROM fact_gw_player f
JOIN dim_team dt_home ON dt_home.team_sk = f.team_sk
JOIN dim_team dt_opp  ON dt_opp.team_sk  = f.opponent_team_sk
WHERE f.season_id = {season_id}
GROUP BY f.team_sk, f.gw
```

Pivot to a matrix of team (rows) × GW (columns). Chart: `px.imshow` with a green→red
diverging colorscale (1=green easy, 5=red hard), with opponent team abbreviation as cell
annotation text. Hover shows full opponent name and H/A.

Note: this shows **played** GWs only — the DB contains only historical data. Display a caption:
"Future fixture data requires a live FPL API call via `etl/fetch.py`."

### Captain Candidates Card

Displayed as a highlighted banner **above** the main table. Shows the top 3 players by predicted
pts across all positions, formatted as cards using `st.columns(3)`:

```python
top3 = df.nlargest(3, f"pred_{model}")[["web_name", "position", "team", f"pred_{model}"]]
for col, row in zip(st.columns(3), top3.itertuples()):
    col.metric(label=f"{row.web_name} ({row.position})", value=f"{row.pred_ridge:.1f} pts", delta=row.team)
```

These are the recommended captain candidates for the GW.

### Main Table

Load predictions for the selected GW via `load_predictions(gw, season_id)`.
`load_predictions` joins player names and team names from the DB (and opponent team via the
SQL described in utils.py).

Displayed columns:

| Column | Source | Notes |
|--------|--------|-------|
| Player | `web_name` (dim_player join) | |
| Position | `position` (from CSV) | |
| Team | team name (dim_team join) | |
| Opponent | opponent team name + H/A suffix | fact_gw_player join; `was_home` flag |
| FDR | derived from `opponent_season_rank` | Fixture Difficulty Rating: rank 1–6 → 1 (easy), 7–12 → 2, 13–16 → 3, 17–18 → 4, 19–20 → 5. Displayed as coloured badge using `st.column_config.NumberColumn` with green→red gradient |
| Price (£m) | `value_lag1 / 10` | Round to 1 dp |
| Predicted pts | `pred_{model}` column | Varies by model selector |
| Ownership % | `selected` resolved from fact_gw_player | `selected` is raw count; divide by total FPL players (~10M) to get %. Join via `(season_id, gw, player_code)` from DB |
| Differential | derived | `st.column_config.CheckboxColumn` — True if predicted pts ≥ position mean AND ownership < 10% |
| Uncertainty (±) | `pred_bayesian_ridge_std` | Only shown when bayesian_ridge selected; requires pred_std addition to predict.py |
| Actual pts | `total_points` | Only populated after GW results; show "—" if NaN |

**Differential filter:** when "Show differentials only" checkbox is ticked in the sidebar, filter
the table to rows where `differential == True`. These are the highest-value low-ownership picks.

Sort by predicted pts descending, then apply top-N filter per position.

Use `st.dataframe(df, use_container_width=True)` with `column_config` for number formatting.

### Section C — Ownership vs Predicted Pts Bubble Chart

The differential finder made visual. Plotted below the main table as a contextual scatter.

```python
fig = px.scatter(
    df,
    x="ownership_pct",
    y=f"pred_{model}",
    size="price_m",
    color="position",
    hover_data=["player", "team", "opponent", "fdr"],
    labels={"ownership_pct": "Ownership %", f"pred_{model}": "Predicted pts"},
)
# Add quadrant reference lines at median ownership and median predicted pts
fig.add_hline(y=df[f"pred_{model}"].median(), line_dash="dot", line_color="grey")
fig.add_vline(x=df["ownership_pct"].median(),  line_dash="dot", line_color="grey")
```

Four quadrants labelled with annotations:
- Top-left: **Differentials** (low ownership, high pts) — transfer in
- Top-right: **Template picks** (high ownership, high pts) — must-have
- Bottom-left: **Avoid** (low ownership, low pts)
- Bottom-right: **Ownership trap** (high ownership, low pts) — consider selling

### Download

```python
st.download_button("Download CSV", df.to_csv(index=False), file_name=f"gw{gw}_s{season}_predictions.csv")
```

---

## Page 5 — Player Scouting (`5_Player_Scouting.py`)

**Purpose:** Value pick identification and rotation risk.

### Section A — Points Boom/Bust Quadrant

The most insightful player characterisation chart in the dashboard. For each player in the
selected season, compute mean and standard deviation of GW points. Plot as a scatter with 4
labelled quadrants — directly answers: "Should I captain this player or just hold them?"

```sql
SELECT
    dp.web_name         AS player,
    dps.position_label  AS position,
    dt.team_name        AS team,
    ROUND(AVG(f.total_points), 2)   AS mean_pts,
    ROUND(stdev(f.total_points), 2) AS std_pts,   -- SQLite stdev via GROUP_CONCAT workaround
    COUNT(DISTINCT f.fixture_id)    AS appearances,
    ROUND(dps.start_cost / 10.0, 1) AS price_m
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND f.minutes > 0
  AND ({position} = 'All' OR dps.position_label = '{position}')
GROUP BY f.player_code
HAVING appearances >= 5
```

Note: SQLite lacks a native `STDDEV` function. Compute in Python post-query:
```python
df_raw = query_db(...)  # returns per-player total_points arrays
# OR: load from fact_gw_player into pandas and compute std per player_code
std_df = df_raw.groupby("player_code")["total_points"].agg(["mean","std","count"]).reset_index()
```

Chart: `px.scatter(x="std_pts", y="mean_pts", color="position", size="price_m",
hover_data=["player", "team", "appearances"])`.

Add reference lines at median mean_pts and median std_pts. Label quadrants with annotations:

| Quadrant | Meaning | FPL implication |
|----------|---------|-----------------|
| High mean, Low SD | Reliable Banker | Captain / hold through blanks |
| High mean, High SD | Boom/Bust | Great captain, risky bench spot |
| Low mean, Low SD | Budget Filler | Safe bench fodder |
| Low mean, High SD | Liability | Sell immediately |

This view immediately shows why picking a consistent mid-priced player can beat a volatile
premium one for non-captain slots.

### Section B — Value Picks (Price vs Predicted Points)

From the loaded prediction CSV, compute:
```python
df["pts_per_million"] = df["pred_ridge"] / (df["value_lag1"] / 10)
```

Scatter chart: `px.scatter(x="value_lag1/10", y="pred_ridge", color="position", hover_data=["web_name", "pts_per_million"])`.
Colour points by `pts_per_million` using a colour scale. Annotate the top-3 value picks per
position with player names.

Companion table: top-5 value picks per position sorted by `pts_per_million`.

### Section B — Form vs Price

Line chart of `pts_rolling_5gw` vs `value_lag1 / 10` (from prediction CSV). Filter by position.
This identifies in-form players at each price point.

### Section C — Player Comparison

Select 2–4 players and overlay their rolling form and season trajectory side by side.

Controls:
- Up to 4 player name search inputs (same partial-match + selectbox pattern as DB Explorer
  Template 4) — rendered in `st.columns(4)`
- `st.selectbox("Metric")` — "pts_rolling_5gw", "pts_rolling_3gw", "season_pts_per_gw_to_date"
- `st.selectbox("Season")` — defaults to current season

Data source: query `fact_gw_player` for each selected `player_code` within the chosen season:
```sql
SELECT gw, {metric}, player_code
FROM fact_gw_player
WHERE season_id = {season_id}
  AND player_code IN ({codes})
  AND minutes > 0
ORDER BY gw
```
Join player names for the legend. Chart: `px.line(x="gw", y=metric, color="player_name",
markers=True)` — all players on the same axes for direct comparison.

### Section D — Price Trajectory

"Is this player rising or falling in price?" — `value` column in `fact_gw_player` tracks
the player's FPL price (£0.1m units) at each GW.

Player selector (same search as Section C). Season defaults to current.

```sql
SELECT gw, ROUND(value / 10.0, 1) AS price_m
FROM fact_gw_player
WHERE player_code = {player_code}
  AND season_id   = {season_id}
  AND minutes > 0   -- or remove to include bench appearances
ORDER BY gw
```

Chart: `px.line(x="gw", y="price_m")` with markers. Annotate start and current price.
Add a second y-axis with `total_points` per GW as a bar chart to correlate price movements
with performance.

### Section E — Component Model Sub-predictions (partially unblocked)

`pred_component_model` and `pred_minutes_model` are present in `logs/training/cv_preds_{pos}.parquet`
(confirmed: 14,240 rows for MID with both columns populated). These OOF predictions can be used
to show historical sub-prediction analysis without touching `run_gw.py`.

**Historical component breakdown (from OOF parquet):**

Load `cv_preds_{pos}.parquet`, join player names, and show a ranked table of OOF component
predictions for the most recent fold (fold=3, season 9→10 predictions):

```python
oof = load_oof(position)
fold3 = oof[oof["fold"] == 3].copy()
fold3["player"] = fold3["player_code"].map(player_names)
# component_model predictions are the combined output; we can't decompose sub-components
# without inspecting the ComponentModel internals, but pred_component_model vs pred_ridge
# difference shows where the component model adds/subtracts value
fold3["component_edge"] = fold3["pred_component_model"] - fold3["pred_ridge"]
```

Show a scatter of `pred_component_model` vs `pred_ridge`, coloured by `component_edge`.
Players where the component model rates significantly higher than Ridge are likely to have
specific goal/assist upside the Ridge model misses.

**P(starts) from minutes_model (from OOF parquet):**

`pred_minutes_model` gives the minutes prediction. Normalise to [0,1] as P(starts) proxy:
```python
fold3["p_starts"] = (fold3["pred_minutes_model"] / 90).clip(0, 1)
```
Show as a sortable table: player, P(starts)%, predicted pts (ridge), team. Low P(starts) +
high ridge prediction = rotation risk flag.

**For current GW (live):** Display `st.info("Live component predictions available after
adding component_model and minutes_model to run_gw.py default model set.")` alongside the
historical OOF analysis.

### Section F — Rotation Risk (deferred)

`minutes_model` P(starts) column is similarly not in the current prediction CSV. Same deferral
pattern as Section C.

---

## Page 6 — Database Explorer (`6_Database_Explorer.py`)

**Purpose:** A frontend for the SQLite DB — browse raw data, run preset parameterised queries
(team rosters, top scorers, season summaries), and execute arbitrary SQL. Covers use cases like:
"Which players were at Man City in 2017-18?", "Who was the top FPL scorer in 2019?", "Show me
all GK scores in GW 10 of 2022-23."

**Data scope note:** The DB contains FPL (Fantasy Premier League) data — fantasy points, player
stats per GW, team rosters per season. It does not contain official Premier League standings
(league table positions, goals in PL sense). Queries like "who won the Prem" are answered via
team aggregate FPL points or squad data, not official PL table data.

---

### Section A — Preset Queries

Templates are organized into four categories to keep the UI navigable as the count grows.
Use two-level selection:

```python
category = st.selectbox("Category", ["Player", "Team", "Gameweek", "Advanced"])
template  = st.selectbox("Query template", TEMPLATES[category])
```

`TEMPLATES` is a dict mapping each category to a list of template name strings. Selecting a
template reveals its parameter inputs below.

**All preset results include:**
- `st.dataframe(result, use_container_width=True)` with native column sorting
- Row count caption: `st.caption(f"{len(result):,} rows")`
- `st.download_button("Download CSV", ...)` beneath every result table
- Era warning banner (via `st.warning`) on any template that uses xG columns when the
  selected season < 7 (pre-2022-23)

---

**Category: Player**

**Category: Player**

#### Template 1: Team Season Roster

"Which players were at [team] in [season]?"

Parameters:
- `st.selectbox("Season")` — populated from `SELECT DISTINCT season_name FROM dim_season ORDER BY season_id`
- `st.selectbox("Team")` — populated from `SELECT DISTINCT team_name FROM dim_team WHERE season_id = {selected_season_id} ORDER BY team_name`
- `st.multiselect("Position")` — GK, DEF, MID, FWD (default: all)

Query:
```sql
SELECT
    dp.web_name         AS player,
    dps.position_label  AS position,
    dps.total_points    AS season_pts,
    ROUND(dps.start_cost / 10.0, 1) AS start_price_m,
    ROUND(dps.end_cost   / 10.0, 1) AS end_price_m,
    dps.appearances,
    dps.goals_scored,
    dps.assists
FROM dim_player_season dps
JOIN dim_player dp   ON dp.player_code  = dps.player_code
JOIN dim_team   dt   ON dt.season_id    = dps.season_id
                    AND dt.team_sk      = dps.team_sk
WHERE dt.team_name   = '{team}'
  AND dps.season_id  = {season_id}
  AND dps.position_label IN ({positions})
ORDER BY dps.total_points DESC
```

#### Template 2: Top FPL Scorers by Season

"Who had the highest FPL points in [season]?"

Parameters:
- `st.selectbox("Season")`
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Show top N", default=20)`

Query:
```sql
SELECT
    dp.web_name         AS player,
    dt.team_name        AS team,
    dps.position_label  AS position,
    dps.total_points    AS season_pts,
    dps.appearances,
    dps.goals_scored,
    dps.assists,
    ROUND(dps.start_cost / 10.0, 1) AS start_price_m
FROM dim_player_season dps
JOIN dim_player dp ON dp.player_code = dps.player_code
JOIN dim_team   dt ON dt.team_sk     = dps.team_sk
WHERE dps.season_id = {season_id}
  AND ({position} = 'All' OR dps.position_label = '{position}')
ORDER BY dps.total_points DESC
LIMIT {n}
```

**Category: Team**

#### Template 3: Team Season Summary

"How did teams compare in [season]?" — proxy for "who won the league" using aggregate FPL data.

Parameters:
- `st.selectbox("Season")`
- `st.selectbox("Sort by")` — Total FPL pts, Goals scored, Appearances

Query:
```sql
SELECT
    dt.team_name                    AS team,
    COUNT(DISTINCT dps.player_code) AS squad_size,
    SUM(dps.total_points)           AS total_fpl_pts,
    SUM(dps.goals_scored)           AS total_goals,
    SUM(dps.assists)                AS total_assists,
    SUM(dps.appearances)            AS total_appearances
FROM dim_player_season dps
JOIN dim_team dt ON dt.team_sk  = dps.team_sk
WHERE dps.season_id = {season_id}
GROUP BY dt.team_name
ORDER BY total_fpl_pts DESC
```

Note displayed beneath result: "Sorted by total FPL points across all squad members — not official
Premier League standing."

#### Template 4: Player Career Stats

"Show me [player]'s career across all seasons."

Parameters:
- `st.text_input("Player name")` → query `dim_player` for partial `web_name` match
- `st.selectbox` to pick from matched players

Query:
```sql
SELECT
    ds.season_name      AS season,
    dt.team_name        AS team,
    dps.position_label  AS position,
    dps.total_points    AS season_pts,
    dps.appearances,
    dps.goals_scored,
    dps.assists,
    ROUND(dps.start_cost / 10.0, 1) AS start_price_m,
    ROUND(dps.end_cost   / 10.0, 1) AS end_price_m
FROM dim_player_season dps
JOIN dim_season ds ON ds.season_id = dps.season_id
JOIN dim_team   dt ON dt.team_sk   = dps.team_sk
WHERE dps.player_code = {player_code}
ORDER BY dps.season_id
```

**Category: Gameweek**

#### Template 5: GW Results

"Show all scores for GW [N] in [season]."

Parameters:
- `st.selectbox("Season")`
- `st.number_input("Gameweek", min=1, max=38)`
- `st.multiselect("Position")` — default: all
- `st.number_input("Minimum minutes played", default=0)`

Query:
```sql
SELECT
    dp.web_name        AS player,
    dps.position_label AS position,
    dt.team_name       AS team,
    f.total_points     AS pts,
    f.minutes,
    f.goals_scored,
    f.assists,
    f.clean_sheets,
    f.bonus,
    f.was_home
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND f.gw = {gw}
  AND f.minutes >= {min_minutes}
  AND dps.position_label IN ({positions})
ORDER BY f.total_points DESC
```

#### Template 6: Player Head-to-Head

"Compare [player A] vs [player B] across seasons."

Parameters:
- Two player search inputs (same partial-match + selectbox pattern as Template 4)

Query: run Template 4 query for each player; display results side-by-side in two `st.columns`.
Add a combined bar chart: `px.bar(barmode="group", x="season", y="season_pts", color="player")`.

**Category: Advanced**

#### Template 7: xG / Advanced Stats Leaders

"Who had the best xG, xA, or xGI in [season]?" — xG era only (seasons 7–10, 2022-23 onwards).

Parameters:
- `st.selectbox("Season")` — filtered to xG era seasons only
- `st.selectbox("Metric")` — xG, xA, xGI, xGC (GK/DEF only for xGC)
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Show top N", default=20)`
- `st.checkbox("Per 90 mins")` — divide by minutes/90 when checked

Query (example for xG total):
```sql
SELECT
    dp.web_name                         AS player,
    dt.team_name                        AS team,
    dps.position_label                  AS position,
    ROUND(SUM(f.expected_goals), 2)     AS total_xg,
    ROUND(SUM(f.expected_assists), 2)   AS total_xa,
    ROUND(SUM(f.expected_goal_involvements), 2) AS total_xgi,
    SUM(f.goals_scored)                 AS goals,
    SUM(f.assists)                      AS assists,
    SUM(f.minutes)                      AS minutes
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND f.minutes > 0
  AND ({position} = 'All' OR dps.position_label = '{position}')
GROUP BY f.player_code
ORDER BY total_xg DESC
LIMIT {n}
```

When "Per 90" is checked, wrap the aggregate column as `ROUND(SUM(...) / (SUM(f.minutes) / 90.0), 3)`.
Display a warning banner if season < 7: "xG stats not available before 2022-23."

---

#### Template 8: Price Movers

"Which players rose or fell the most in price in [season]?"

Parameters:
- `st.selectbox("Season")`
- `st.selectbox("Direction")` — "Biggest risers", "Biggest fallers", "Both"
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Show top N", default=20)`

Query:
```sql
SELECT
    dp.web_name         AS player,
    dt.team_name        AS team,
    dps.position_label  AS position,
    ROUND(dps.start_cost / 10.0, 1)                        AS start_price_m,
    ROUND(dps.end_cost   / 10.0, 1)                        AS end_price_m,
    ROUND((dps.end_cost - dps.start_cost) / 10.0, 1)       AS price_change_m,
    dps.total_points                                        AS season_pts
FROM dim_player_season dps
JOIN dim_player dp ON dp.player_code = dps.player_code
JOIN dim_team   dt ON dt.team_sk     = dps.team_sk
WHERE dps.season_id = {season_id}
  AND dps.start_cost IS NOT NULL
  AND dps.end_cost   IS NOT NULL
  AND ({position} = 'All' OR dps.position_label = '{position}')
ORDER BY price_change_m {'DESC' if direction == 'Biggest risers' else 'ASC'}
LIMIT {n}
```

Companion bar chart: `px.bar(x="player", y="price_change_m", color="position")`.

---

#### Template 9: Reliable Starters (Minutes Leaders)

"Who played the most minutes in [season]?" — rotation risk signal for GK/DEF selection.

Parameters:
- `st.selectbox("Season")`
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Min appearances", default=10)`
- `st.number_input("Show top N", default=20)`

Query:
```sql
SELECT
    dp.web_name                                     AS player,
    dt.team_name                                    AS team,
    dps.position_label                              AS position,
    SUM(f.minutes)                                  AS total_minutes,
    COUNT(DISTINCT f.fixture_id)                    AS appearances,
    ROUND(AVG(f.minutes), 1)                        AS avg_mins_per_gw,
    ROUND(SUM(f.total_points) * 1.0
          / COUNT(DISTINCT f.fixture_id), 2)        AS pts_per_app,
    dps.total_points                                AS season_pts
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND f.minutes > 0
  AND ({position} = 'All' OR dps.position_label = '{position}')
GROUP BY f.player_code
HAVING appearances >= {min_appearances}
ORDER BY total_minutes DESC
LIMIT {n}
```

---

#### Template 10: Transfer & Ownership Trends

"Who was most transferred in/out in GW [N] of [season]?"

Note: `transfers_in`, `transfers_out`, and `selected` (ownership count) are in `fact_gw_player`.
Available for all seasons but most meaningful for the current/recent season.

Parameters:
- `st.selectbox("Season")`
- `st.number_input("Gameweek")`
- `st.selectbox("Metric")` — "Transfers in", "Transfers out", "Ownership (selected)"
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Show top N", default=20)`

Query (example for transfers_in):
```sql
SELECT
    dp.web_name         AS player,
    dt.team_name        AS team,
    dps.position_label  AS position,
    f.transfers_in,
    f.transfers_out,
    f.selected          AS ownership,
    f.total_points      AS pts_that_gw,
    ROUND(f.value / 10.0, 1) AS price_m
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND f.gw = {gw}
  AND ({position} = 'All' OR dps.position_label = '{position}')
ORDER BY f.{metric_col} DESC
LIMIT {n}
```

`metric_col` maps "Transfers in" → `transfers_in`, "Transfers out" → `transfers_out`,
"Ownership" → `selected`.

---

#### Template 11: Defensive Record (Clean Sheets & Goals Conceded)

"Which teams kept the most clean sheets in [season]?" — fixture difficulty context for GK/DEF planning.

Parameters:
- `st.selectbox("Season")`
- `st.selectbox("Metric")` — "Clean sheet rate", "Goals conceded (team level)"
- Note displayed: "Team-level goals conceded is derived from match scores, not from the
  player-level `goals_conceded` column (which is time-on-pitch scoped and unreliable for
  team totals)."

Query:
```sql
SELECT
    dt.team_name,
    COUNT(DISTINCT f.fixture_id)                        AS fixtures_played,
    SUM(CASE WHEN f.clean_sheets = 1 THEN 1 ELSE 0 END) AS gk_clean_sheet_gws,
    ROUND(
        100.0 * SUM(CASE WHEN f.clean_sheets = 1 THEN 1 ELSE 0 END)
        / COUNT(DISTINCT f.fixture_id), 1
    )                                                   AS cs_pct
FROM fact_gw_player f
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team dt ON dt.team_sk = f.team_sk
WHERE f.season_id = {season_id}
  AND dps.position_label = 'GK'
  AND f.minutes >= 60
GROUP BY dt.team_name
ORDER BY cs_pct DESC
```

Uses GK rows with 60+ minutes as a proxy for team clean sheets — avoids the known bias in
the player-level `goals_conceded` column documented in CLAUDE.md.

---

#### Template 12: Bonus Point Leaders

"Who accumulated the most bonus points in [season]?"

`bonus` (points awarded 1–3 per match) and `bps` (raw bonus point score) are in `fact_gw_player`
for all seasons.

Parameters:
- `st.selectbox("Season")`
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Show top N", default=20)`

Query:
```sql
SELECT
    dp.web_name                             AS player,
    dt.team_name                            AS team,
    dps.position_label                      AS position,
    SUM(f.bonus)                            AS total_bonus,
    ROUND(AVG(f.bonus), 2)                  AS avg_bonus_per_gw,
    SUM(f.bps)                              AS total_bps,
    COUNT(DISTINCT f.fixture_id)            AS appearances,
    dps.total_points                        AS season_pts,
    ROUND(100.0 * SUM(f.bonus)
          / dps.total_points, 1)            AS bonus_pct_of_pts
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND f.minutes > 0
  AND ({position} = 'All' OR dps.position_label = '{position}')
GROUP BY f.player_code
ORDER BY total_bonus DESC
LIMIT {n}
```

`bonus_pct_of_pts` shows how bonus-dependent a player's score is — useful for identifying
players whose FPL returns may be inflated or deflated by BPS system volatility.

#### Template 13: Haul Hunters (High-Score GW Frequency)

"Which players score 10+ points most often?" — the primary captain selection signal.

Parameters:
- `st.selectbox("Season")` — "All xG era seasons" option + individual seasons
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Haul threshold (pts)", default=10, min=6, max=20)`
- `st.number_input("Show top N", default=20)`

Query:
```sql
SELECT
    dp.web_name                                             AS player,
    dt.team_name                                            AS team,
    dps.position_label                                      AS position,
    COUNT(DISTINCT f.fixture_id)                            AS appearances,
    SUM(CASE WHEN f.total_points >= {threshold} THEN 1 ELSE 0 END) AS hauls,
    ROUND(
        100.0 * SUM(CASE WHEN f.total_points >= {threshold} THEN 1 ELSE 0 END)
        / COUNT(DISTINCT f.fixture_id), 1
    )                                                       AS haul_rate_pct,
    ROUND(AVG(f.total_points), 2)                           AS avg_pts,
    MAX(f.total_points)                                     AS best_gw,
    dps.total_points                                        AS season_pts
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}     -- or omit for all xG era
  AND f.minutes > 0
  AND ({position} = 'All' OR dps.position_label = '{position}')
GROUP BY f.player_code
HAVING appearances >= 5
ORDER BY hauls DESC, haul_rate_pct DESC
LIMIT {n}
```

Companion bar chart: `px.bar(x="player", y="hauls", color="position", hover_data=["haul_rate_pct", "avg_pts"])`.

---

#### Template 14: Current Form Table

"Who is in form over the last [N] gameweeks?" — primary weekly transfer-in signal.

Parameters:
- `st.selectbox("Season")` — defaults to latest (season 10)
- `st.number_input("Last N gameweeks", default=5, min=1, max=10)`
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Min minutes per GW", default=45)`
- `st.number_input("Show top N", default=30)`

Implementation: first determine `max_gw` for the selected season, then filter to the last N GWs.

```sql
WITH max_gw AS (
    SELECT MAX(gw) AS mgw FROM fact_gw_player WHERE season_id = {season_id}
)
SELECT
    dp.web_name                             AS player,
    dt.team_name                            AS team,
    dps.position_label                      AS position,
    COUNT(DISTINCT f.fixture_id)            AS gws_played,
    SUM(f.total_points)                     AS total_pts,
    ROUND(AVG(f.total_points), 2)           AS avg_pts,
    MAX(f.total_points)                     AS best_gw,
    ROUND(AVG(f.minutes), 0)               AS avg_minutes,
    ROUND(f.value / 10.0, 1)               AS current_price_m
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk, max_gw
WHERE f.season_id = {season_id}
  AND f.gw > max_gw.mgw - {n_gws}
  AND f.minutes >= {min_minutes}
  AND ({position} = 'All' OR dps.position_label = '{position}')
GROUP BY f.player_code
ORDER BY total_pts DESC
LIMIT {n}
```

`current_price_m` uses the most recent `value` column from `fact_gw_player` (£0.1m units ÷ 10).

---

#### Template 15: Home vs Away Player Splits

"How does [player] perform at home vs away?"

Parameters:
- Player name search (same partial-match + selectbox pattern as Template 4)
- `st.selectbox("Season")` — "All seasons" or specific

Query:
```sql
SELECT
    CASE WHEN f.was_home = 1 THEN 'Home' ELSE 'Away' END    AS venue,
    COUNT(DISTINCT f.fixture_id)                             AS appearances,
    SUM(f.total_points)                                      AS total_pts,
    ROUND(AVG(f.total_points), 2)                            AS avg_pts,
    SUM(f.goals_scored)                                      AS goals,
    SUM(f.assists)                                           AS assists,
    SUM(f.clean_sheets)                                      AS clean_sheets,
    ROUND(AVG(f.minutes), 0)                                AS avg_minutes,
    -- xG columns only when season >= 7:
    ROUND(SUM(f.expected_goals), 2)                          AS xg,
    ROUND(SUM(f.expected_assists), 2)                        AS xa
FROM fact_gw_player f
WHERE f.player_code = {player_code}
  AND ({season_id} IS NULL OR f.season_id = {season_id})
  AND f.minutes > 0
GROUP BY venue
```

Display side-by-side with `st.columns(2)`. Add a bar chart comparing avg_pts home vs away
across seasons: `px.bar(barmode="group", x="season", y="avg_pts", color="venue")`.

---

#### Template 16: Attacking Returns (Goals + Assists Leaders)

"Who contributed the most goals and assists in [season]?" — separate from FPL pts ranking,
for pure attacking output scouting.

Parameters:
- `st.selectbox("Season")`
- `st.selectbox("Sort by")` — "Goals", "Assists", "Goal involvements (G+A)", "xGI (xG era only)"
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Min appearances", default=5)`
- `st.number_input("Show top N", default=20)`

Query:
```sql
SELECT
    dp.web_name                                     AS player,
    dt.team_name                                    AS team,
    dps.position_label                              AS position,
    SUM(f.goals_scored)                             AS goals,
    SUM(f.assists)                                  AS assists,
    SUM(f.goals_scored) + SUM(f.assists)            AS goal_involvements,
    ROUND(SUM(f.expected_goals), 2)                 AS xg,
    ROUND(SUM(f.expected_assists), 2)               AS xa,
    ROUND(SUM(f.expected_goal_involvements), 2)     AS xgi,
    COUNT(DISTINCT f.fixture_id)                    AS appearances,
    dps.total_points                                AS season_pts
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND f.minutes > 0
  AND ({position} = 'All' OR dps.position_label = '{position}')
GROUP BY f.player_code
HAVING appearances >= {min_appearances}
ORDER BY {sort_col} DESC
LIMIT {n}
```

xG/xA/xGI columns displayed as NULL for seasons < 7 with a `st.warning` banner.

---

**Category: Team**

(Templates 3 and 11 belong here — already documented above.)

---

**Category: Gameweek**

(Templates 5 and 10 belong here — already documented above.)

#### Template 17: Double Gameweek Finder

"Which players had two fixtures in GW [N] of [season]?"

Double gameweeks occur when a player has 2 rows in `fact_gw_player` for the same
`(season_id, gw)` but different `fixture_id` values.

Parameters:
- `st.selectbox("Season")`
- `st.selectbox("GW")` — auto-populated; DGW GWs can be detected automatically
  (highlight GWs where any player has > 1 fixture)
- `st.selectbox("Position")` — "All" + individual positions

Implementation: auto-detect DGW gameweeks on page load for the selected season:
```sql
SELECT DISTINCT gw
FROM fact_gw_player
WHERE season_id = {season_id}
GROUP BY player_code, gw
HAVING COUNT(DISTINCT fixture_id) > 1
ORDER BY gw
```
Populate the GW selector only with detected DGW gameweeks. If none exist, show
`st.info("No double gameweeks detected in this season.")`.

Main query:
```sql
SELECT
    dp.web_name                         AS player,
    dt.team_name                        AS team,
    dps.position_label                  AS position,
    COUNT(DISTINCT f.fixture_id)        AS fixtures,
    SUM(f.total_points)                 AS total_pts,
    SUM(f.minutes)                      AS total_minutes,
    SUM(f.goals_scored)                 AS goals,
    SUM(f.assists)                      AS assists,
    ROUND(f.value / 10.0, 1)           AS price_m
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND f.gw = {gw}
  AND ({position} = 'All' OR dps.position_label = '{position}')
GROUP BY f.player_code
HAVING fixtures > 1
ORDER BY total_pts DESC
```

---

**Category: Advanced**

(Templates 7, 8, 9, 12 belong here — already documented above.)

#### Template 18: GK-Specific Stats

"Who were the best performing goalkeepers in [season]?" — GK scoring is driven by
saves, clean sheets, and penalty saves, not goals/assists.

Parameters:
- `st.selectbox("Season")`
- `st.selectbox("Sort by")` — "Total FPL pts", "Saves", "Clean sheets", "Save pts per GW"
- `st.number_input("Min appearances", default=10)`
- `st.number_input("Show top N", default=20)`

Query:
```sql
SELECT
    dp.web_name                                     AS player,
    dt.team_name                                    AS team,
    COUNT(DISTINCT f.fixture_id)                    AS appearances,
    SUM(f.saves)                                    AS total_saves,
    ROUND(AVG(f.saves), 1)                          AS avg_saves_per_gw,
    ROUND(SUM(f.saves) / 3.0, 0)                   AS save_pts,
    SUM(f.clean_sheets)                             AS clean_sheets,
    ROUND(
        100.0 * SUM(f.clean_sheets)
        / COUNT(DISTINCT f.fixture_id), 1
    )                                               AS cs_rate_pct,
    SUM(f.goals_conceded)                           AS goals_conceded_approx,
    SUM(f.penalties_saved)                          AS pen_saves,
    SUM(f.bonus)                                    AS total_bonus,
    dps.total_points                                AS season_pts,
    ROUND(dps.start_cost / 10.0, 1)               AS start_price_m
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND dps.position_label = 'GK'
  AND f.minutes >= 60
GROUP BY f.player_code
HAVING appearances >= {min_appearances}
ORDER BY {sort_col} DESC
LIMIT {n}
```

Note: `goals_conceded` is time-on-pitch scoped per player (see CLAUDE.md known quirk) — displayed
with a `st.caption` noting it is not the team's match total.

---

#### Template 19: Suspension Risk (Card Accumulation)

"Which players are at risk of suspension?" — yellow/red card tracking for transfer planning.

Parameters:
- `st.selectbox("Season")` — defaults to current season (10)
- `st.selectbox("Position")` — "All" + individual positions
- `st.number_input("Show top N", default=30)`

Query:
```sql
SELECT
    dp.web_name                             AS player,
    dt.team_name                            AS team,
    dps.position_label                      AS position,
    SUM(f.yellow_cards)                     AS yellow_cards,
    SUM(f.red_cards)                        AS red_cards,
    COUNT(DISTINCT f.fixture_id)            AS appearances,
    ROUND(
        100.0 * SUM(f.yellow_cards)
        / COUNT(DISTINCT f.fixture_id), 1
    )                                       AS yellow_per_100_gws,
    dps.total_points                        AS season_pts
FROM fact_gw_player f
JOIN dim_player        dp  ON dp.player_code  = f.player_code
JOIN dim_player_season dps ON dps.player_code = f.player_code
                           AND dps.season_id  = f.season_id
JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
WHERE f.season_id = {season_id}
  AND f.minutes > 0
  AND ({position} = 'All' OR dps.position_label = '{position}')
GROUP BY f.player_code
ORDER BY yellow_cards DESC, red_cards DESC
LIMIT {n}
```

Highlight rows where `yellow_cards >= 5` (typical Premier League suspension threshold) using
`st.dataframe` with `column_config` — no direct row colouring in Streamlit, but a derived
`at_risk` boolean column can be added and filtered.

---

#### Template 20: Season History Explorer

"Show a player's full FPL history, including seasons before the current DB scope."

Uses the `fact_player_season_history` table — this is the only table containing cross-season
history pulled directly from the FPL API `element-summary` endpoint, and it covers seasons
outside the main fact_gw_player scope.

Parameters:
- Player name search (same partial-match + selectbox pattern as Template 4)

Query:
```sql
SELECT
    ds.season_name                          AS season,
    dt.team_name                            AS team,
    h.total_points,
    h.appearances,
    h.goals_scored,
    h.assists,
    ROUND(h.start_cost / 10.0, 1)          AS start_price_m,
    ROUND(h.end_cost   / 10.0, 1)          AS end_price_m,
    ROUND((h.end_cost - h.start_cost) / 10.0, 1) AS price_change_m
FROM fact_player_season_history h
JOIN dim_season ds ON ds.season_id  = h.season_id
LEFT JOIN dim_team dt ON dt.team_sk = h.team_sk
WHERE h.player_code = {player_code}
ORDER BY h.season_id
```

Display as a table + `px.bar(x="season", y="total_points")` career points chart. This
complements Template 4 (Player Career Stats from dim_player_season) by covering earlier
seasons and surfacing FPL API history data not otherwise visible in the dashboard.

---

### Section B — Table Browser

Raw table access for power users who want to see the underlying data.

Controls:
- `st.selectbox("Table")` — `dim_player`, `dim_team`, `dim_season`, `dim_player_season`,
  `fact_gw_player`, `fact_player_season_history`
- `st.number_input("Rows to show", default=50, max=500)`
- Dynamic column filters: after the table is selected, render one `st.text_input` per column
  for substring/range filtering

Implementation: build a `WHERE` clause from non-empty filter inputs; use parameterised queries.
Display result as `st.dataframe(use_container_width=True)` with a row count below.

---

### Section C — Free-Form SQL

For arbitrary queries not covered by presets.

```python
sql = st.text_area("SQL query", height=150, placeholder="SELECT * FROM dim_player LIMIT 10")
if st.button("Run"):
    try:
        result = utils.query_db(sql)
        st.dataframe(result, use_container_width=True)
        st.caption(f"{len(result):,} rows returned")
        st.download_button("Download CSV", result.to_csv(index=False), file_name="query_result.csv")
    except Exception as e:
        st.error(str(e))
```

Safety note: the DB is local and read-only for this purpose. No write operations are possible
from the dashboard (SQLite file opened read-only via `sqlite3.connect(DB_PATH, uri=True)` with
`?mode=ro` flag in `utils.query_db`).

---

### Section D — Schema Reference

Collapsible `st.expander("DB Schema")` showing the 6-table star schema summary:

| Table | Grain | Key columns |
|-------|-------|-------------|
| `dim_season` | season | `season_id`, `season_name`, `total_gws` |
| `dim_player` | player | `player_code`, `web_name`, `first_name`, `second_name` |
| `dim_team` | team × season | `team_sk`, `team_name`, `season_id` |
| `dim_player_season` | player × season | `player_code`, `season_id`, `team_sk`, `total_points`, `position_label` |
| `fact_gw_player` | player × fixture | `player_code`, `season_id`, `gw`, `fixture_id`, `total_points`, `minutes` |
| `fact_player_season_history` | player × prior season | `player_code`, `season_id`, `total_points`, `start_cost`, `end_cost` |

This lets users write their own SQL queries without needing to consult external docs.

---

The Streamlit entry point. Displays:
- Project title and current GW/season (auto-detected from `list_available_gws()`)
- Last monitoring run date (from `monitoring_log.csv` max `logged_at`)
- Summary metrics card: latest GW MAE per position vs threshold (4-column layout)
- Navigation guide: brief description of each page
- Link to `project_plan.md` rendered inline (optional)

---

## predict.py Change — Add pred_std

Add the following to `ml/predict.py` in the model prediction loop, when `bayesian_ridge` is
an active model:

```python
# Inside the per-position prediction block
if "bayesian_ridge" in active_models:
    br_model = get_model("bayesian_ridge", pos)
    pred_mean, pred_std = br_model.predict(X_scaled, return_std=True)
    preds_df["pred_bayesian_ridge"]     = pred_mean
    preds_df["pred_bayesian_ridge_std"] = pred_std
```

This adds `pred_bayesian_ridge_std` to the output CSV, enabling the uncertainty band display
on Page 4.

---

## Static Chart Audit

All required static charts already exist — no new generation step needed:

**`outputs/eda/` (24 PNGs — all present):**
- `points_distribution.png`, `home_away_effect.png`, `team_strength_heatmap.png`
- `missing_data_matrix.png`, `top6_fixture_effect.png`, `price_vs_season_points.png`
- `price_band_performance.png`, and 17 others

**`outputs/models/` (17 PNGs — all present):**
- `calibration_{GK,DEF,MID,FWD}.png`
- `mae_by_fold_{GK,DEF,MID,FWD}.png`
- `shap_{GK,DEF,MID,FWD}.png`
- `residuals_{GK,DEF,MID,FWD}.png`
- `learning_curves.png`

If any chart is regenerated (e.g., after retraining), the dashboard picks it up automatically
on next page load via `st.image()` with `use_column_width=True` — no dashboard code changes needed.

---

## Implementation Steps

Each step is independently deliverable and testable. Complete phases A–B first to get a
working dashboard as fast as possible; later phases add depth.

### Phase A — Foundation (no pages yet; sets up scaffolding)

| Step | Task | Files created / modified | Notes |
|------|------|--------------------------|-------|
| A1 | Install dependencies | `requirements.txt` (new) | Add `streamlit>=1.35`, `plotly>=5.20` |
| A2 | Streamlit config | `outputs/dashboards/.streamlit/config.toml` (new) | Sets `gatherUsageStats=false`; `layout="wide"` set per-page via `st.set_page_config` |
| A3 | Create directory skeleton | `outputs/dashboards/`, `outputs/dashboards/pages/` | `mkdir` only; no code yet |
| A4 | `utils.py` — path constants + `query_db` | `outputs/dashboards/utils.py` (new) | `PROJECT_ROOT`, `DB_PATH`, all dir constants; read-only SQLite URI; parameterised queries |
| A5 | `utils.py` — player/team name loaders | `outputs/dashboards/utils.py` | `load_player_names()`, `load_team_names()` with `@st.cache_data` |
| A6 | `utils.py` — prediction and monitoring loaders | `outputs/dashboards/utils.py` | `list_available_gws()`, `load_predictions()` (with opponent join), `load_monitoring_log()` |
| A7 | `utils.py` — CV and OOF loaders | `outputs/dashboards/utils.py` | `load_cv_metrics()`, `load_oof(position)` from `cv_preds_{pos}.parquet`; joins player names |
| A8 | `app.py` landing page | `outputs/dashboards/app.py` (new) | `st.set_page_config(layout="wide")`; monitoring summary metric cards (latest GW MAE × 4 positions); navigation guide |
| A9 | `ml/predict.py` — add `pred_std` | `ml/predict.py` | Add `return_std=True` for bayesian_ridge; write `pred_bayesian_ridge_std` column to CSV |

**Gate:** `streamlit run outputs/dashboards/app.py` launches with landing page showing GW 30 monitoring cards.

---

### Phase B — GW Predictions page (highest weekly value)

| Step | Task | Files created / modified | Notes |
|------|------|--------------------------|-------|
| B1 | Fixture Difficulty Calendar | `pages/4_GW_Predictions.py` (new) | Team × GW `px.imshow` heatmap; FDR 1–5 colour scale; annotation text = opponent abbreviation |
| B2 | Captain Candidates card | `pages/4_GW_Predictions.py` | Top-3 predicted pts as `st.metric` cards in `st.columns(3)` above the table |
| B3 | Sidebar controls | `pages/4_GW_Predictions.py` | GW selector, model selector, position multiselect, price slider, top-N, differentials checkbox |
| B4 | Main predictions table | `pages/4_GW_Predictions.py` | All columns: player, position, team, opponent (H/A), FDR badge, price, predicted pts, ownership %, differential flag, uncertainty ±, actual pts |
| B5 | Ownership bubble chart + download | `pages/4_GW_Predictions.py` | 4-quadrant `px.scatter` (template / differential / trap / avoid); `st.download_button` |

**Gate:** Page 4 loads for GW 30, table shows 287 rows with player names and FDR badges, captain candidates card shows top 3, bubble chart renders.

---

### Phase C — Database Explorer (20 templates)

| Step | Task | Files created / modified | Notes |
|------|------|--------------------------|-------|
| C1 | Navigation structure + schema reference | `pages/6_Database_Explorer.py` (new) | Two-level `st.selectbox` (category → template); collapsible schema expander; per-result download button and row count pattern established |
| C2 | Player category templates (1, 2, 4, 6) | `pages/6_Database_Explorer.py` | Team Roster, Top Scorers, Player Career, Head-to-Head |
| C3 | Player category templates (13, 14, 15, 16) | `pages/6_Database_Explorer.py` | Haul Hunters, Current Form Table, Home/Away Splits, Attacking Returns |
| C4 | Team category templates (3, 11) | `pages/6_Database_Explorer.py` | Team Season Summary, Defensive Record |
| C5 | Gameweek category templates (5, 10, 17) | `pages/6_Database_Explorer.py` | GW Results, Transfer & Ownership, Double GW Finder (with auto-detection) |
| C6 | Advanced category templates (7, 8, 9, 12) | `pages/6_Database_Explorer.py` | xG Leaders (per-90 toggle), Price Movers, Reliable Starters, Bonus Leaders |
| C7 | Advanced category templates (18, 19, 20) | `pages/6_Database_Explorer.py` | GK Stats, Suspension Risk, Season History Explorer |
| C8 | Table browser + free-form SQL | `pages/6_Database_Explorer.py` | Table selector with dynamic column filters; SQL text area with error handling; read-only enforced |

**Gate:** All 20 templates return results. Man City 2017-18 roster query works. DGW auto-detection identifies correct GWs. Free-form SQL `SELECT * FROM dim_player LIMIT 5` returns table with download button.

---

### Phase D — Model Performance page

| Step | Task | Files created / modified | Notes |
|------|------|--------------------------|-------|
| D1 | CV comparison table | `pages/3_Model_Performance.py` (new) | Load `cv_metrics_all.csv`; mean across folds; ridge row highlighted; position tabs via `st.tabs` |
| D2 | Interactive OOF calibration | `pages/3_Model_Performance.py` | `load_oof()` from utils; model selector + fold selector; `px.scatter` pred vs actual with hover player labels; x=y reference line |
| D3 | Static diagnostic plots | `pages/3_Model_Performance.py` | `st.image` for SHAP, MAE-by-fold, learning curves; 2×2 grid with `st.columns` |
| D4 | Residual decomposition | `pages/3_Model_Performance.py` | Join OOF parquet with `fact_gw_player`; bar charts by home/away, opponent tier, price band, minutes bucket |
| D5 | Monitoring trend + eval report viewer | `pages/3_Model_Performance.py` | Rolling MAE line chart with threshold lines; GW report selectbox rendering markdown inline |

**Gate:** CV table loads with ridge highlighted. Interactive scatter shows hover labels for GW 30 players. Residual bar charts show mean residual per bucket. GW 30 eval report renders inline.

---

### Phase E — Data Explorer page

| Step | Task | Files created / modified | Notes |
|------|------|--------------------------|-------|
| E1 | Points histogram + home/away bar | `pages/1_Data_Explorer.py` (new) | `px.histogram` with position colour + season facet; `px.bar` home/away mean pts |
| E2 | Team strength heatmap + attack/defence scatter | `pages/1_Data_Explorer.py` | `px.imshow` goals-conceded heatmap; `px.scatter` attack vs defence with quadrant lines |
| E3 | Player career trajectory | `pages/1_Data_Explorer.py` | Partial name search → selectbox → `px.line` pts/GW per season |
| E4 | xG vs actual scatter + era comparison | `pages/1_Data_Explorer.py` | `px.scatter` with x=y ref line; embed `era_comparison.png` |

**Gate:** Points histogram filters by season. Searching "Salah" shows career line chart. xG scatter renders for MID season 8 with hover labels.

---

### Phase F — Bias & Data Quality page

| Step | Task | Files created / modified | Notes |
|------|------|--------------------------|-------|
| F1 | Bias markdown + feature availability + price embeds | `pages/2_Bias_Quality.py` (new) | `st.markdown` for data_biases.md; `st.image` for missing_data_matrix, top6_fixture_effect, price PNGs |
| F2 | Known Data Quirks table | `pages/2_Bias_Quality.py` | Hard-coded 7-row DataFrame with `st.warning` banner; `st.dataframe` |

**Gate:** Page renders all static images and quirks table without errors.

---

### Phase G — Player Scouting page

| Step | Task | Files created / modified | Notes |
|------|------|--------------------------|-------|
| G1 | Boom/bust quadrant | `pages/5_Player_Scouting.py` (new) | DB query for mean/std pts per player; `px.scatter` with 4 labelled quadrants; position + season filters |
| G2 | Value picks scatter + form vs price | `pages/5_Player_Scouting.py` | `pts_per_million` computed from prediction CSV; top-5 value tables per position |
| G3 | Player comparison + price trajectory | `pages/5_Player_Scouting.py` | Multi-player `px.line` overlay; single-player price line with pts bar on second axis |
| G4 | Component model (OOF historical) | `pages/5_Player_Scouting.py` | Load `cv_preds_{pos}.parquet` fold 3; component edge scatter; `p_starts` from `pred_minutes_model / 90` |

**Gate:** Boom/bust quadrant populates for MID season 10. Selecting Salah + Haaland shows overlaid pts_rolling_5gw. Component edge scatter renders for fold 3 DEF.

---

### Phase H — Integration & Polish

| Step | Task | Files created / modified | Notes |
|------|------|--------------------------|-------|
| H1 | Error/empty state testing | All page files | Temporarily rename a prediction CSV; confirm warning shown not traceback on Page 4 |
| H2 | Cache performance check | All page files | Confirm `@st.cache_data` prevents re-queries on page re-run (no perceptible delay on second load) |
| H3 | End-to-end verification | — | Run all 15 verification checks listed in the Verification section below |

---

## Verification

After implementation, verify end-to-end:

1. **Start dashboard:** `streamlit run outputs/dashboards/app.py` from the project root — no errors on startup.
2. **Landing page:** Monitoring summary shows GW 30 Season 10 data; all 4 positions PASS.
3. **Page 4 — GW 30:** Select GW 30, model = ridge. Table shows 287 rows with player names and opponent teams. Download CSV button produces a valid file.
4. **Page 4 — bayesian_ridge:** Uncertainty column (pred_bayesian_ridge_std) appears after running `python -m ml.predict --gw 30 --season 10 --models bayesian_ridge` to regenerate the prediction CSV with the pred_std addition.
5. **Page 3:** CV table loads; ridge row highlighted as best MAE per position. Position selector switches images correctly.
6. **Page 1 — Player search:** Enter a player name (e.g. "Salah") → select from matches → line chart renders across seasons.
7. **Page 5 — Value picks:** pts_per_million scatter populates; top-5 value tables show sensible players.
8. **Page 4 — FDR and differentials:** FDR column shows coloured badges 1–5. "Show differentials only" checkbox filters to low-ownership high-predicted players. Captain candidates card shows top 3 above the table.
9. **Page 5 — Player comparison:** Select Salah + Haaland → rolling pts overlay chart renders. Price trajectory for one player shows GW-by-GW price with pts bars on second axis.
10. **Page 1 — xG scatter:** Select MID, season 8 → scatter of xG vs actual goals with x=y reference line and player hover labels.
11. **Page 2 — Known Quirks:** Table of 7 quirks with warning banner renders correctly.
12. **Page 3 — Eval reports:** GW selector populates from `logs/monitoring/`; selecting GW 30 renders the narrative markdown inline.
13. **Page 6 — Team roster (Template 1):** Select category "Player" → "Team Season Roster", choose 2017-18, Man City → table shows all Man City players with FPL pts and prices, CSV download present. **DGW finder (Template 17):** Select a known DGW season/GW → only players with 2 fixtures appear. Free-form SQL box accepts `SELECT * FROM dim_player LIMIT 5` and returns results with download button.
14. **Empty states:** Delete a prediction CSV; Page 4 shows the warning message, not a traceback.
15. **Caching:** On repeated page visits, `@st.cache_data` prevents repeated DB queries (verify with no delay on re-render).
