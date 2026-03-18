# Phase 7 — Interactive Dashboard Report

## Summary

Phase 7 implements a 6-page Streamlit dashboard that surfaces all pipeline outputs
(predictions, CV metrics, monitoring, historical data, model diagnostics) in one
interactive local application.

**Launch command:**
```
streamlit run outputs/dashboards/app.py
```

---

## Files Delivered

| File | Role |
|------|------|
| `outputs/dashboards/app.py` | Landing page: monitoring summary cards + latest predictions tabs |
| `outputs/dashboards/utils.py` | Shared data loaders, path constants, `query_db()` helper |
| `outputs/dashboards/.streamlit/config.toml` | Headless config, no usage stats, light theme |
| `outputs/dashboards/pages/1_Data_Explorer.py` | Historical EDA: distributions, home/away, team heatmap, career trajectories, xG scatter |
| `outputs/dashboards/pages/2_Bias_Quality.py` | Bias reference: ML biases doc, schema eras, fixture difficulty, price vs performance, known quirks |
| `outputs/dashboards/pages/3_Model_Performance.py` | CV table, OOF calibration scatter, static diagnostics, monitoring trend, residual decomposition, eval report viewer |
| `outputs/dashboards/pages/4_GW_Predictions.py` | FDR calendar, prediction table with filters, captain cards, ownership bubble chart, CSV download |
| `outputs/dashboards/pages/5_Player_Scouting.py` | Boom/bust quadrant, value picks scatter, player comparison, price trajectory, component model analysis |
| `outputs/dashboards/pages/6_Database_Explorer.py` | 20 SQL templates across 4 categories, table browser, free-form SQL, schema reference |

---

## Page-by-page Specification

### Landing Page (app.py)
- 4 metric cards: latest GW MAE per position vs threshold
- Tabs per position: top-10 predicted players from most recent GW CSV
- Navigation guide table

### Page 1 — Data Explorer
- Sidebar: season multiselect (default xG era 7–10), position filter
- Section A: GW points distribution histogram faceted by season
- Section B: Home vs away mean pts grouped bar
- Section C: Team strength heatmap (goals conceded from match scores — not from player `goals_conceded` column)
- Section D: Player career trajectory — partial name search → line chart by season + summary table
- Section E: xG vs actual goals scatter (xG era only) with x=y reference line
- Section F: Era comparison static PNG embed
- Section G: Team attack vs defence scatter with quadrant labels and median reference lines

### Page 2 — Bias & Data Quality
- Full `docs/data_biases.md` rendered inline
- Schema era summary table (hardcoded markdown, 6 eras)
- Missing data matrix PNG with era restriction caption
- Top-6 fixture effect PNG + interactive opponent rank vs pts bar chart from feature matrix parquets
- Price vs performance dual-image layout
- Known data quirks dataframe (7 rows) with warning banner

### Page 3 — Model Performance
- CV comparison table: mean MAE/RMSE/Spearman across folds; ridge highlighted, best MAE per cell highlighted yellow
- OOF calibration scatter: pred vs actual with hover (player, GW), x=y reference, Pearson r + MAE summary
- Static diagnostic plots: MAE-by-fold, SHAP, calibration, residuals, learning curves
- Monitoring trend: rolling MAE line + threshold dashes + alert markers
- Residual decomposition: home/away, opponent tier, price band, minutes bucket bar charts (via OOF-to-feature-matrix join)
- Per-GW eval report viewer: selectbox from `logs/monitoring/gw*_s*_eval.md` files, inline markdown render

### Page 4 — GW Predictions
- FDR calendar heatmap (from feature matrix `opponent_season_rank`, opponent abbreviations as cell text)
- Captain candidate metric cards (top 3 by pred_ridge)
- Prediction table: player, position, team, opponent (H/A), FDR badge, price, predicted pts, ownership%, differential flag, uncertainty (pred_bayesian_ridge_std), actual pts
- Ownership bubble chart: 4-quadrant scatter (Differentials / Template / Avoid / Trap)
- CSV download button

### Page 5 — Player Scouting
- Boom/bust quadrant: mean vs std of GW pts per player (std computed in pandas — SQLite lacks STDDEV)
- Value picks scatter: pts_per_million (pred_ridge / price_m), top-3 annotated, top-5 tables per position
- Form vs price scatter: pts_rolling_5gw vs price
- Player comparison: up to 4 players, pts_rolling_5gw and pts_rolling_3gw computed in pandas from raw data
- Price trajectory: dual-axis go.Figure — price line + pts bars per GW, start/end annotations
- Component model OOF: component_edge scatter, rotation risk table (p_starts = pred_minutes_model / 90)

### Page 6 — Database Explorer
20 SQL templates across 4 categories:

**Player:** Team Roster (T1), Top Scorers (T2), Career Stats (T4), H2H (T6), Haul Hunters (T13), Form Table (T14), Home/Away Splits (T15), Attacking Returns (T16)

**Team:** Season Summary (T3), Defensive Record via GK proxy (T11)

**Gameweek:** GW Results (T5), Transfers (T10), DGW Finder auto-detecting double fixtures (T17)

**Advanced:** xG Leaders with per-90 toggle (T7), Price Movers (T8), Reliable Starters (T9), Bonus Leaders (T12), GK Stats (T18), Suspension Risk (T19), Season History (T20)

Additionally: table browser with column filters, free-form SQL editor with error handling, collapsible schema reference.

---

## Implementation Notes

### Key fixes applied during implementation

| Issue | Fix |
|-------|-----|
| `opponent_season_rank` not in `fact_gw_player` | Sourced from feature matrix parquet in `load_fdr_calendar()` |
| `applymap` deprecated in pandas 2.3.3 | Changed to `.map()` throughout |
| `dim_player_season.appearances` column absent | Computed via `COUNT(DISTINCT fixture_id)` |
| `dim_season` uses `season_label` not `season_name` | Fixed throughout all queries |
| Manager "AM" rows in seasons 9–10 | Added `AND position_label IN ('GK','DEF','MID','FWD')` filter |
| SQLite lacks STDDEV | Boom/bust std computed in pandas post-query |
| Rolling metrics not in `fact_gw_player` | Computed in pandas from raw `total_points` |
| `fact_player_season_history` has no `team_sk` | Removed team join from Template 20 |

### pred_bayesian_ridge_std addition (ml/predict.py)
A `_predict_bayesian_ridge_std()` helper was added to `ml/predict.py`. It calls
`model.predict(return_std=True)` on the BayesianRidge model after applying the same
imputation and scaling steps as the main prediction path. The std column appears in the
prediction CSV as `pred_bayesian_ridge_std` when the `bayesian_ridge` model is included in
the model set (default for `run_gw.py`).

### Shared utils layer (utils.py)
All pages import from `utils.py` via `sys.path.insert`. The module provides:
- `query_db(sql, params)` — read-only SQLite connection via `file:...?mode=ro` URI
- `load_predictions(gw, season_id)` — reads CSV, joins web_name + team + opponent
- `load_fdr_calendar(season_id)` — derives FDR from feature matrix `opponent_season_rank` (1–6 → FDR 5, 19–20 → FDR 1)
- `load_oof(position)` — reads `logs/training/cv_preds_{pos}.parquet`
- `load_monitoring_log()`, `load_cv_metrics()`, `load_season_list()`, etc.
- All loaders decorated with `@st.cache_data`

---

## Integration Check Results

All 21 pre-launch checks passed:

- DB accessible (247,308 rows)
- GW 30 predictions loaded (287 rows)
- Required columns present: web_name, pred_ridge, value_lag1, pts_rolling_5gw
- OOF parquets present for all 4 positions
- Feature matrices present for all 4 positions
- Monitoring log populated (8 rows)
- CV metrics populated (660 rows)
- FDR calendar loads (600 rows)
- Season list loads (10 seasons)
- data_biases.md present (resolved via `__file__` at page runtime)
- Empty GW returns empty DataFrame (graceful, no exception)
- All 3 required EDA static PNGs present

HTTP 200 confirmed on local Streamlit launch (port 8501).

---

## Plan Alignment

All items from `phase7_plan.md` implemented. See that file for the full specification.
