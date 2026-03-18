# Phase 9 Monitoring Report

## 1. What Is Operational

### 1.1 Per-GW metrics logging

After each gameweek result is published, `run_gw.py` runs the monitoring step
automatically (Step 4). For each position (GK, DEF, MID, FWD) it computes:

| Metric | Description |
|--------|-------------|
| MAE | Mean absolute error between predicted and actual total_points |
| RMSE | Root mean squared error |
| Spearman rho | Rank correlation between predicted and actual scores |
| Top-10 precision | Fraction of true top-10 scorers captured in predicted top-10 |

Results are appended to `logs/monitoring/monitoring_log.csv`.

### 1.2 5-GW rolling MAE and alert thresholds

After each GW, the 5-GW rolling MAE is computed per position per model.
An alert flag (`alert=1`) is set when rolling MAE exceeds the position threshold.

Thresholds are 1.5 x the CV baseline MAE (rolling-mean baseline, Phase 6):

| Position | Baseline MAE (CV mean) | Alert threshold (1.5x) |
|----------|----------------------:|----------------------:|
| GK | 2.329 | 3.494 |
| DEF | 2.332 | 3.498 |
| MID | 1.997 | 2.996 |
| FWD | 2.406 | 3.609 |

Alerts trigger a WARNING log line. No automatic retraining is initiated -- the
operator reviews the rolling trend before deciding to retrain.

### 1.3 Current monitoring status

| GW | GK MAE | DEF MAE | MID MAE | FWD MAE | Alerts |
|----|--------|---------|---------|---------|--------|
| 24 | 2.184 | 2.003 | 2.068 | 2.235 | none |
| 30 | 1.988 | 2.735 | 1.742 | 1.825 | none |

Both evaluated GWs are well within threshold. No alerts raised to date.

---

## 2. Schema Change Alerting (Phase 9 new)

### 2.1 Design

Before each ETL rebuild, `run_gw.py` calls `_step_schema_check()`, which
compares the actual column set of `merged_gw.csv` against `EXPECTED_COLS`
in `etl/schema.py`. Alerts are written to `logs/monitoring/schema_alerts.csv`.

The check runs even when `--skip-fetch` is passed (always inspects the
current CSV before ETL runs). It does NOT abort the pipeline -- schema
changes are flagged for operator review but processing continues.

Two alert types:

| check_type | Meaning |
|------------|---------|
| `new_columns` | FPL added a column not present in EXPECTED_COLS |
| `dropped_columns` | A column in EXPECTED_COLS is absent from the CSV |

### 2.2 Expected column sets per season

The `EXPECTED_COLS` dict in `etl/schema.py` is built from era flags in `SEASONS`:

| Era | Seasons | Column additions |
|-----|---------|-----------------|
| Old Opta | 1-4 (2016-17 to 2019-20) | Core only |
| Modern core | 5-6 (2020-21, 2021-22) | + xP |
| xG era | 7-8 (2022-23, 2023-24) | + xG/xA/xGI/xGC, starts |
| Manager era | 9 (2024-25) | + mng_* |
| Defensive era | 10 (2025-26) | + clearances_blocks_interceptions, defensive_contribution, recoveries, tackles, starts, mng_* (NULL columns retained in CSV) |

When FPL adds a new column group in a future season, the operator should:
1. Add the new season row to `SEASONS` in `etl/schema.py` with correct era flags
2. Add the column frozenset constant (e.g. `_NEW_COLS`) and update the loop
3. Run `python -m etl.run` to validate the new schema

### 2.3 Schema alerts log

**File:** `logs/monitoring/schema_alerts.csv`

**Schema:** `season_id, gw, check_type, columns, logged_at`

The file is initialised with a header row only. Rows are appended by
`_step_schema_check()` whenever a mismatch is detected.

---

## 3. Per-GW Narrative Summaries (Phase 9 new)

### 3.1 Design

At the end of each `run_gw.py` monitoring step, `_write_gw_eval_report()`
produces a markdown file with a structured narrative for operator review.

**Output path:** `logs/monitoring/gw{N}_s{season}_eval.md`

### 3.2 Report structure

| Section | Content |
|---------|---------|
| Summary | Per-position MAE, RMSE, Spearman, Top-10 Prec, Rolling MAE, Alert flag |
| Top Predictions vs Actuals | Top 5 by predicted score + top 5 actual scorers per position |
| Largest Misses | Top 5 highest absolute errors across all positions |
| Rolling Trend | Last 5 GWs MAE history per position (from monitoring_log.csv) |
| Alert Status | PASS/ALERT per position with threshold values |

### 3.3 Example output

`logs/monitoring/gw30_s10_eval.md` was generated on 2026-03-18 for GW 30
(season 10). All four positions showed PASS. DEF had highest MAE (2.74)
driven by several 11-point haul defenders (clean sheet + goal combinations)
that were not predicted.

---

## 4. Dynamic CV Folds

The CV fold definitions in `ml/evaluate.py` are now computed dynamically
from `etl/schema.py SEASONS` rather than hardcoded:

```python
_XG_SEASON_IDS = sorted(s[0] for s in SEASONS if s[7] == 1)
CV_FOLDS = [
    (_XG_SEASON_IDS[:i], _XG_SEASON_IDS[i])
    for i in range(1, len(_XG_SEASON_IDS))
]
```

Current folds (4 xG era seasons: 7, 8, 9, 10):

| Fold | Train | Validate |
|------|-------|----------|
| 1 | 2022-23 | 2023-24 |
| 2 | 2022-23, 2023-24 | 2024-25 |
| 3 | 2022-23, 2023-24, 2024-25 | 2025-26 |

When season 11 is added to `SEASONS` with `has_xg_stats=1`, fold 4
(`[7,8,9,10] -> 11`) is added automatically -- no code changes required.

---

## 5. End-of-Season Retraining Procedure

The `retrain_season.py` script at the repo root orchestrates all retraining
steps. Run it after a new season's data is complete and `etl/schema.py` has
been updated.

### 5.1 Prerequisites (manual steps before running)

1. Add new season row to `SEASONS` in `etl/schema.py` (season_id, label,
   start/end year, total_gws, era flags, team_map_source)
2. Add season label to `_SEASON_LABELS` dict in `run_gw.py`
3. Add season to `SEASON_NAME_TO_ID` in `etl/schema.py`
4. Confirm `data/{new_season}/gws/merged_gw.csv` is complete (all GWs fetched)
5. Confirm `data/{new_season}/players_raw.csv` is present

### 5.2 Execution

```bash
python retrain_season.py --season 11
```

**Steps executed:**

| Step | Action |
|------|--------|
| 1 Verify | Validate prerequisites and data directories |
| 2 Archive | Copy models/*.pkl and *_meta.json to models/v{prev_season}/ |
| 3 ETL | Rebuild db/fpl.db (python -m etl.run) |
| 4 Cache | Delete outputs/features/*.parquet (stale feature matrices) |
| 5 Evaluate | Run python -m ml.evaluate (all positions, new CV folds) |
| 6 Alpha | Run python -m ml.train --model ridge --alpha-search |
| 7 Train | Run python -m ml.train --all (all 22 tabular + decomposed models) |
| 8 Meta | Run python -m ml.train --meta (simple_avg, stacking, blending) |
| 9 Report | Write logs/training/retrain_s{season}_report.md |

**Flags:**

| Flag | Effect |
|------|--------|
| `--skip-archive` | Skip Step 2 (already archived) |
| `--skip-etl` | Skip Step 3 (already rebuilt) |
| `--dry-run` | Print all steps without executing any |

**Dry-run verification:**

```bash
python retrain_season.py --season 11 --dry-run
```

Prints all 9 steps and their intended actions without touching any files.

### 5.3 After retraining

- Review `logs/training/retrain_s{season}_report.md` for new CV MAE
- Compare against previous-season thresholds; update alert thresholds in
  `run_gw.py _THRESHOLDS` if baseline MAE has shifted by more than 5%
- Run `python run_gw.py --gw 1 --season {N}` for the first GW of the new
  season to confirm the pipeline is operational end-to-end
- Consider running Optuna LightGBM tuning if FWD MAE is above baseline:
  `python -m ml.train --tune --position FWD`

---

## 6. Optuna LightGBM Tuning

The `--tune` flag in `ml/train.py` is fully implemented. FWD is highest
priority based on Phase 6 evaluation results.

```bash
python -m ml.train --tune --position FWD   # highest priority
python -m ml.train --tune                  # all positions
python -m ml.train --meta                  # rebuild meta-models after tuning
```

Tuned hyperparameters are stored in `models/{position}_lgbm_meta.json`
under the `lgbm_params` field. Optuna tuning has not yet been run as of
Phase 9 completion; Ridge remains the production model.

---

## 7. Monitoring Log Schema

**File:** `logs/monitoring/monitoring_log.csv`

| Column | Type | Description |
|--------|------|-------------|
| season_id | int | Season identifier (10 = 2025-26) |
| gw | int | Gameweek number |
| model | str | Model name (e.g. 'ridge') |
| position | str | Position group (GK/DEF/MID/FWD) |
| mae | float | Mean absolute error for this GW |
| rmse | float | Root mean squared error |
| spearman | float | Spearman rank correlation |
| top10_precision | float | Fraction of true top-10 in predicted top-10 |
| rolling_mae_5gw | float | 5-GW rolling mean MAE |
| threshold | float | Alert threshold (1.5x baseline MAE) |
| alert | int | 1 if rolling_mae_5gw > threshold, else 0 |
| logged_at | str | ISO 8601 UTC timestamp |

---

## 8. Phase 7 (Dashboard) Implications

The monitoring outputs are designed to feed into the Phase 7 interactive
dashboard. Recommended surfaces:

| Monitoring output | Dashboard element |
|-------------------|------------------|
| `monitoring_log.csv` rolling MAE | Line chart: model performance trend by position |
| Per-GW alert flags | Status indicator / alert banner per position |
| Top-10 precision | Bar chart comparing positions over time |
| `gw{N}_eval.md` largest misses | Table: biggest prediction errors this GW |
| Schema alerts | Admin panel: column drift warnings |

The dashboard should allow filtering by position and model, and should
highlight any GW where `alert=1` was raised.

---

## 9. Files Delivered (Phase 9)

| File | Type | Description |
|------|------|-------------|
| `etl/schema.py` | Modified | Added EXPECTED_COLS dict + era column sets |
| `run_gw.py` | Modified | Added _step_schema_check(), _write_gw_eval_report(); wired into run() |
| `ml/evaluate.py` | Modified | Dynamic CV_FOLDS from SEASONS; dynamic FOLD_LABELS |
| `retrain_season.py` | New | End-of-season retraining orchestrator (9 steps) |
| `logs/monitoring/schema_alerts.csv` | New | Schema drift log (header row, no alerts) |
| `docs/monitoring_report.md` | New | This file |
