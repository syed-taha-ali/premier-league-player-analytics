# Code Review — FPL Analysis Pipeline

**Reviewed:** 2026-03-18
**Reviewer:** Claude Code (claude-sonnet-4-6)
**Scope:** All Python source files and the EDA notebook across all 9 phases.

---

## Executive Summary

This is a well-engineered, end-to-end data science pipeline with clear phase boundaries,
solid data integrity practices, and thoughtful design throughout. The schema design, CV
strategy, and monitoring layer are particular strengths. The issues identified are mostly
minor — hardcoded constants, one off-by-one error in a comment, a misleading comment in the
stacking logic, and a few style inconsistencies — rather than structural problems. The
pipeline is close to production-ready.

Overall rating: **8.5 / 10**

---

## 1. Project Structure and Architecture

### Strengths

The project is well-organised with clear phase-based separation:

```
etl/        — data ingestion and transformation
ml/         — feature engineering, training, evaluation, inference
outputs/    — cached artefacts (feature matrices, predictions, charts)
logs/       — training metrics, monitoring log, eval reports
models/     — serialised model bundles
data/       — raw CSVs (read-only for historical seasons)
```

Each layer has a single entry point (`etl/run.py`, `ml/features.py`,
`ml/train.py`, `ml/evaluate.py`, `ml/predict.py`, `run_gw.py`), making the
execution path easy to trace.

The registry pattern in `ml/models.py` (`ModelSpec` dataclass) is a professional
design choice: new models are registered once and automatically participate in CV,
training, serialisation, and inference without touching the loop logic in `evaluate.py`
or `train.py`.

### Issues

**1.1 — Hardcoded current season in `run_gw.py` and `retrain_season.py`**

`run_gw.py` line 433 defaults `season_id` to `10`:

```python
def run(gw: int | None = None, season_id: int = 10, ...):
```

`_SEASON_LABELS` in `run_gw.py` (lines 55–59) and `_step_etl` comment in
`retrain_season.py` (line 284) also reference season 10 explicitly. When season 11
arrives, these defaults become misleading without a code change.

**Suggestion:** Derive the current season from `etl.schema.SEASONS` at import time.

```python
from etl.schema import SEASONS as _SEASONS
_CURRENT_SEASON: int = max(s[0] for s in _SEASONS)
```

Then replace the hardcoded `10` default and the static `_SEASON_LABELS` dict with
`{s[0]: s[1] for s in _SEASONS}`.

---

**1.2 — `os.path` vs `pathlib.Path` style split in `ml/features.py`**

`features.py` is the only module in the `ml/` package that uses `os.path` throughout:

```python
# features.py (lines 29-31)
_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, '..', 'db', 'fpl.db')
```

Every other module (`evaluate.py`, `train.py`, `predict.py`, `run_gw.py`) uses
`pathlib.Path`. This is a minor inconsistency but makes `features.py` harder to read
alongside its neighbours.

**Suggestion:** Replace with:

```python
_HERE = Path(__file__).parent.parent
DB_PATH = str(_HERE / 'db' / 'fpl.db')
OUTPUTS_DIR = str(_HERE / 'outputs' / 'features')
```

---

**1.3 — `print()` vs `logging` inconsistency in `features.py`**

`features.py` uses bare `print()` for progress messages (lines 284, 287, 302), while
every other module uses a `log = logging.getLogger(__name__)` pattern. This means
feature-build messages bypass any log-level filters and cannot be suppressed.

**Suggestion:** Add a module-level logger and replace `print()` with `log.info()`.

---

**1.4 — `print()` in monitoring summary in `run_gw.py`**

`_step_monitor` (line 278) uses `print()` for the monitoring summary table, breaking
the otherwise consistent use of `log.*` in that module.

---

## 2. Data Engineering Pipeline

### Strengths

- The star schema is correctly designed with stable cross-season `player_code` as the
  bridge key. The `fpl_id`-resets-per-season quirk is documented and handled correctly.
- The `goals_conceded` exclusion note (CLAUDE.md and `features.py` module docstring) is
  a good example of surfacing a non-obvious data quality issue at the point of use.
- `_TEAM_MATCH_SQL` derives team goals from score columns rather than `goals_conceded`,
  which avoids the pitch-time scoping problem entirely.
- The COVID gap (2019-20 GWs 30–38) and GW7 2022-23 absence are documented and handled
  by the data model (not patched over).
- Double Gameweek rows are handled correctly: `fixture_id` is included in the fact
  table grain.

### Issues

**2.1 — No database backup before ETL rebuild**

`etl/run.py` (line 46) deletes `fpl.db` before rebuilding it:

```python
db_path = _HERE / 'db' / 'fpl.db'
if db_path.exists():
    db_path.unlink()
```

If the rebuild fails partway (e.g., mid-season data corruption), the previous valid
database is gone. Unlike models, which `retrain_season.py` archives before overwriting,
there is no equivalent backup for the database.

**Suggestion:** Rename the existing file to `fpl.db.bak` before rebuild, and restore it
on failure:

```python
bak = db_path.with_suffix('.db.bak')
if db_path.exists():
    db_path.rename(bak)
try:
    _build_db(db_path)
    if bak.exists():
        bak.unlink()
except Exception:
    if bak.exists():
        bak.rename(db_path)
    raise
```

---

**2.2 — ETL validation count comment mismatch in `run_gw.py`**

`run_gw.py` line 158 reads:

```python
log.info('[step2] ETL complete — all 11 validation checks passed')
```

`etl/validate.py` `run_all()` docstring and the code itself contain exactly **10**
checks (numbered 1–10). The comment is off by one.

**Fix:** Change `11` to `10`.

---

**2.3 — No minimum GW count guard in `retrain_season.py`**

`_validate_prerequisites()` verifies that `merged_gw.csv` exists but does not check
that the new season has a sufficient number of GWs to form a valid training fold. If
retraining is triggered too early in the season (e.g., after only 3 GWs), the CV will
produce misleading metrics or fail silently.

**Suggestion:** Add a row-count or max-GW check:

```python
df_gw = pd.read_csv(csv_path, usecols=['GW'])
max_gw = df_gw['GW'].max()
if max_gw < 20:
    raise RuntimeError(
        f'Season {season} only has {max_gw} GWs in merged_gw.csv. '
        f'Retraining requires a complete season (>= 20 GWs recommended).'
    )
```

---

**2.4 — No rollback on partial `retrain_season.py` failure**

If model training (step 7) fails after the archive (step 2), the current `models/`
directory is now empty and the archive is the only copy. The pipeline aborts with a
`RuntimeError`, but there is no recovery step to restore the archive automatically.

**Suggestion:** At minimum, document this in the step 2 log message and add a note to
the `retrain_season.py` CLI help text explaining that `--skip-archive` can be used to
re-run from step 3 without overwriting the archive again.

---

## 3. Code Quality

### Strengths

- Naming is consistent and descriptive across the codebase (`_step_fetch`,
  `_step_etl`, `build_feature_matrix`, `stratified_impute`).
- Docstrings are present on all public and most private functions after the recent
  review pass.
- SQL constants (`_BASE_SQL`, `_OPP_RANK_CTE`, `_TEAM_MATCH_SQL`) are separated from
  function logic and well commented.
- `CONTEXT_COLS`, `TARGET_COL`, `VALID_POSITIONS` are defined once and imported where
  needed, avoiding magic strings.

### Issues

**3.1 — Duplicate `new_df` assignment in `_step_monitor`**

`run_gw.py` lines 252 and 272 both assign `new_df = pd.DataFrame(new_rows)`:

```python
# line 252
new_df = pd.DataFrame(new_rows)

for row in new_rows:
    ...  # mutates row dicts in-place

# line 272 — rebuilds from the now-mutated rows, but looks like a bug
new_df = pd.DataFrame(new_rows)
```

The first assignment at line 252 is stale by line 272 because the `rolling_mae_5gw`
and `alert` fields are filled by the loop. The second build is the correct one. The
first should be removed.

---

**3.2 — Misleading comment on `oof_records` in `evaluate.py`**

Line 444:

```python
oof_records: list[dict] = []   # accumulates tabular val predictions across folds for stacking
```

`oof_records` is only appended to **after** Pass 2 (line 496), meaning the stacking
meta-model in the current fold receives the OOF predictions from **all previous folds**,
not the current one. This is the correct design for unbiased stacking, but the comment
implies it accumulates for the current fold's stacking — which it does not.

**Suggestion:**

```python
# OOF predictions from prior folds — passed to stacking meta-model for unbiased fitting.
# The current fold's OOF is appended after Pass 2 so the stacker never sees its own
# validation predictions during training.
oof_records: list[dict] = []
```

---

**3.3 — Hardcoded `'ridge'` in rolling trend section of `_write_gw_eval_report`**

`run_gw.py` line 397:

```python
hist = hist[hist['model'] == 'ridge']
```

This ignores the `primary_model` argument that was used for all other sections of the
same report. If the pipeline is run with `--primary-model lgbm`, the Rolling Trend
section will still show Ridge data.

**Fix:**

```python
hist = hist[hist['model'] == primary_model]
```

---

**3.4 — `gw_season_ids` hash uses `season_id * 1000 + gw`**

`evaluate.py` lines 397 and 458 compute a GW grouping key as:

```python
gw_keys = val_df['season_id'].values * 1000 + val_df['gw'].values
```

This works with the current data (max GW ~47) but the chosen multiplier (1000) is an
undocumented assumption. A tuple key is both safer and self-documenting.

**Suggestion:** Convert to a structured key in `compute_metrics`:

```python
# Instead of gw_season_ids: np.ndarray, accept a list of tuples or a pandas MultiIndex
```

This is a minor change for correctness; the current code will not produce a collision
in practice.

---

**3.5 — `_parse_args` in `run_gw.py` missing docstring**

Every other `_parse_args` function in the codebase has a one-line docstring
(`retrain_season.py`, `ml/evaluate.py`, `ml/predict.py`). `run_gw.py`'s `_parse_args`
at line 474 is the one exception.

---

## 4. Performance and Scalability

### Strengths

- Parquet caching of feature matrices avoids expensive SQLite queries on repeated runs.
- LightGBM's native NaN handling avoids the imputation overhead for tree-based models.
- `StandardScaler` is fit only on the training fold; `transform` is applied to the
  validation fold. The same fitted scaler is bundled for inference.
- `@st.cache_data` is applied to all dashboard data loaders, preventing repeated DB
  queries across user interactions.

### Issues

**4.1 — Feature matrix built sequentially for four positions**

`build_feature_matrix` is called per position in several places. In `run_gw.py`
`_step_predict` and in `retrain_season.py`, positions are processed sequentially. The
four position matrices share `_TEAM_MATCH_SQL` data but each makes a separate DB
connection and runs the full CTE.

For the current dataset size (~40k rows per matrix build) this is not a bottleneck, but
note the redundancy if scalability becomes relevant.

---

**4.2 — `_add_opponent_features` creates a second copy of team match data**

`features.py` `_add_opponent_features` (line 430) calls `tm = tm.copy()` then
recomputes cumulative and rolling stats on the same source that `_add_team_features`
already computed from. Both functions call `groupby(['team_sk', 'season_id'])` and run
`transform` lambdas on the same underlying rows.

This is correct (both functions need to compute stats from the opponent's perspective),
but it means the same aggregation work is done twice for the team match data. For the
current scale this is not an issue.

---

**4.3 — `iterrows()` in `_write_gw_eval_report`**

`run_gw.py` uses `for _, r in top_pred.iterrows()` (lines 359, 370, 387) to build
markdown table rows. `iterrows()` is significantly slower than vectorised string
formatting for DataFrames, though the call sites here operate on DataFrames of 5 rows
(top-5 predictions), so the performance impact is zero. Worth noting for any future
refactor that uses iterrows on larger DataFrames.

---

## 5. Modelling Practices

### Strengths

- **Temporal CV with expanding windows** — never random splits. This is the correct
  strategy for time-series-structured data and prevents future leakage.
- **Leakage prevention is explicit and documented.** `bonus`, `bps`, `ict_index` are
  called out as banned features in CLAUDE.md and the `features.py` module docstring.
  `shift(1)` is applied consistently before all rolling and lag operations.
- **Imputation is fold-scoped.** `stratified_impute` computes means from the training
  fold only, with season-level stratification and a global fallback. The imputer is
  stored in the model bundle for inference. This is a correct and professional approach.
- **Baseline gate.** Every model is checked against `pts_rolling_5gw` on ≥ 2 of 3
  primary metrics before being considered viable. The gate is automated and its result
  is logged per model per position.
- **OOF meta-model training.** The stacking and blending meta-models are fit on OOF
  predictions from prior folds, preventing the meta-learner from seeing its own
  validation data.
- **Dynamic CV fold construction.** `CV_FOLDS` is derived from `SEASONS` at import time
  in `evaluate.py`. Adding a new season with `has_xg_stats=1` automatically creates a
  fourth fold without any code change.

### Issues

**5.1 — LightGBM fitted without early stopping**

`build_lgbm` (evaluate.py line 292) calls `model.fit(X_train, y_train)` without
passing `eval_set` or `callbacks`. A fixed `n_estimators` (300 for DEF/MID/FWD) is
used regardless of how much data is available. For the GK position (smallest dataset)
this may result in overfitting.

**Suggestion:** Pass `eval_set=[(X_val, y_val)]` and use
`early_stopping_rounds` via a LightGBM callback when validation data is available.
This is especially relevant before Optuna tuning is run, since the baseline
`n_estimators=200` for GK was chosen heuristically.

---

**5.2 — `xgi_rolling_5gw` still in the MID/FWD feature matrix**

CLAUDE.md §8 documents that `xgi_rolling_5gw` was dropped from Ridge for MID/FWD due
to collinearity. The drop happens in `ml/models.py` `_build_ridge` at train time.
However, the feature is still present in `_POSITION_FEATURES['MID']` and
`_POSITION_FEATURES['FWD']` in `features.py` (lines 103 and 125), so it remains in the
parquet cache and is available to all other models (LightGBM, MLP, etc.).

This is intentional for LightGBM (which handles correlated features natively), but the
separation between "feature matrix" and "Ridge-specific feature set" is not visible
without reading both files. A comment at the `_POSITION_FEATURES` definition would
clarify this design choice.

---

**5.3 — Monitoring alert thresholds are hardcoded**

`run_gw.py` (lines 44–49) defines:

```python
_THRESHOLDS = {
    'GK':  3.494,
    'DEF': 3.498,
    'MID': 2.996,
    'FWD': 3.609,
}
```

These are derived from 1.5× the baseline MAE after Phase 6 CV. If CV MAE improves
significantly after retraining with a new season, these thresholds become miscalibrated
(too loose). `retrain_season.py` step 9 includes "Update monitoring alert thresholds if
baseline MAE has shifted significantly" as a manual next-step, but does not automate it.

**Suggestion:** The retrain report at step 9 could compute and print the suggested new
thresholds (1.5 × new baseline MAE) so the operator only needs to copy them.

---

## 6. Error Handling and Robustness

### Strengths

- All subprocess calls in `run_gw.py` and `retrain_season.py` check `returncode` and
  raise `RuntimeError` on failure, preventing silent pipeline continuation.
- `_step_schema_check` is non-fatal — schema changes are logged but do not abort the
  pipeline. This is the right design: FPL's schema may drift, but not every change
  breaks the ETL.
- `fetch.py` implements exponential backoff with a configurable number of retries for
  the FPL API, with distinct handling for critical vs non-critical endpoints.
- ETL `validate.py` raises `AssertionError` on any check failure, causing the ETL
  subprocess to exit non-zero and aborting the GW runner.

### Issues

**6.1 — Points reconciliation check excludes current season, but no audit trail**

`etl/validate.py` check 6 explicitly skips the current (in-progress) season:

```python
WHERE dps.season_id < {current_season}
```

This is correct and well-justified. However, if a data integrity issue arises in the
current season, this check will not catch it until the season completes. There is no
alternative lightweight check for the current season.

**Suggestion:** Add a soft check for the current season that reports divergence at `> 20
pts` (a higher tolerance) as a warning rather than a failure. This allows large FPL
retroactive corrections to pass without blocking the pipeline, while catching obvious
data corruption.

---

**6.2 — Broad `except Exception` in `6_Database_Explorer.py`**

Page 6 of the dashboard wraps the free-form SQL execution in `except Exception as e`.
This is acceptable at the UI boundary where users can type arbitrary SQL and any failure
mode must be surfaced as a friendly error message. The broad catch is appropriate here.

However, `query_db()` in `utils.py` also catches `sqlite3.OperationalError` internally.
This means a OperationalError from a malformed query is caught twice — once returning an
empty DataFrame, once in the UI exception handler. The double-catch is harmless but
could mask the original error message in the UI.

---

**6.3 — FetchError re-raise in `fetch.py` may not propagate the final HTTP error**

`fetch.py`'s retry loop (approximate lines 71–98) raises on the last attempt if
`critical=True`, but the `last_exc` variable is only set inside the `except` block.
If the final attempt returns a non-200 response rather than raising an exception,
`last_exc` may be `None` and the `raise last_exc` at the bottom of the loop would fail
with `TypeError: exceptions must derive from BaseException`.

This is an edge case that requires both the retry logic and the HTTP response status
check to be reached together, but it should be guarded.

---

## 7. Reproducibility

### Strengths

- All random states are fixed (`random_state=42` for Ridge, `seed=42` for LightGBM,
  Optuna TPE sampler seeded with `42`). Results are reproducible given the same data.
- `--dry-run` mode in `retrain_season.py` prints all steps without executing, allowing
  review of the execution plan before committing.
- `requirements.txt` (or equivalent) captures dependencies. The ETL + training pipeline
  can be re-run end-to-end with `python -m etl.run` → `python -m ml.train --all` →
  `python run_gw.py`.
- Feature matrices are cached to parquet but can be force-rebuilt with `force=True`.
  CLAUDE.md documents when the cache must be cleared (after adding new GWs).

### Issues

**7.1 — No explicit Python version or environment lock file visible**

The repository has `requirements.txt` but no `pyproject.toml`, `.python-version`, or
`conda.yml`. Running on Python 3.9 vs 3.12 could produce different behaviour from
`pandas` type inference changes, `sklearn` API changes, or `lightgbm` wheel availability.

**Suggestion:** Add a `.python-version` file (if using pyenv) or pin the Python version
in a `pyproject.toml`.

---

**7.2 — `COALESCE(_opp_rank.season_rank, 10)` is a silent fallback**

`features.py` line 211:

```python
COALESCE(_opp_rank.season_rank, 10) AS opponent_season_rank
```

If a team's opponent rank cannot be computed (e.g., the opponent has no results in
`fact_gw_player` for that season), the rank defaults silently to 10 (mid-table).
This is a reasonable default but is not validated or logged anywhere.

---

## 8. Dashboard and App Layer

### Strengths

- All database connections in `utils.py` use `?mode=ro` (read-only URI), preventing
  any accidental writes from the dashboard.
- `@st.cache_data` is applied to all data-loading functions, ensuring expensive queries
  are not re-run on each UI interaction.
- The shared `utils.py` layer cleanly separates data access from page rendering. All
  pages import via `sys.path.insert`, keeping the data logic in one place.
- SQL injection surface is minimised: user-facing inputs that feed into SQL strings
  are validated against a whitelist of `POSITIONS` before embedding.

### Issues

**8.1 — `pos_filter` SQL embedding uses Python whitelist, not parameterised queries**

`pages/1_Data_Explorer.py` and `pages/5_Player_Scouting.py` embed `pos_filter` /
`bb_pos` directly into SQL f-strings after whitelist validation. For the current
controlled set of four positions this is safe, but parameterised queries are the
correct pattern even for enumerable inputs:

```python
# Current pattern:
sql = f"WHERE position_label = '{pos_filter}'"

# Better:
sql = "WHERE position_label = ?"
df = query_db(sql, params=(pos_filter,))
```

`query_db` in `utils.py` passes `params` to `pd.read_sql`, so the infrastructure
already supports this.

---

**8.2 — FDR calendar uses MID feature matrix as a proxy for all positions**

`utils.py` `load_fdr_calendar()` loads the MID feature matrix to derive opponent rank:

```python
df = load_predictions('MID')   # representative
```

This is correct for fixture difficulty (a team's opponent rank is the same regardless
of whether you are a GK or FWD facing them). The comment "representative" helps, but
explicitly stating the reason would prevent a future reader from assuming this is a
data quality shortcut.

---

**8.3 — `app.py` rank column fallback does not include `pred_blending`**

`app.py` (approximately lines 105–109) tries `pred_ridge`, then `pred_blending`, then
`pred_ensemble` to determine the rank column. `pred_ensemble` is not produced by the
standard pipeline (`run_gw.py` default models are ridge, bayesian_ridge, blending);
`pred_ensemble` only appears if more than one model is requested via `--model`. The
fallback order should end at `pred_blending`, not `pred_ensemble`, or the fallback
should log a warning when none of the expected columns are present.

---

## Top 5 Highest Priority Fixes

These are the issues most likely to cause silent errors or incorrect results in
production.

**Priority 1 — Hardcoded `'ridge'` in rolling trend (`run_gw.py:397`)**

```python
# Current:
hist = hist[hist['model'] == 'ridge']
# Fix:
hist = hist[hist['model'] == primary_model]
```

This silently shows Ridge metrics in the eval report regardless of which model was used
for monitoring, making the rolling trend section misleading when `--primary-model` is
overridden.

---

**Priority 2 — Duplicate `new_df` build in `_step_monitor` (`run_gw.py:252`)**

Line 252 builds `new_df = pd.DataFrame(new_rows)` before the loop that mutates
`row['rolling_mae_5gw']` and `row['alert']` in-place. The resulting `new_df` is
missing both columns. Line 272 correctly rebuilds it after the loop, making line 252
both stale and misleading.

**Fix:** Delete line 252.

---

**Priority 3 — Validation check count mismatch (`run_gw.py:158`)**

```python
# Current:
log.info('[step2] ETL complete — all 11 validation checks passed')
# Fix:
log.info('[step2] ETL complete — all 10 validation checks passed')
```

`validate.py` has 10 checks. The comment is off by one.

---

**Priority 4 — Hardcoded `season_id=10` default (`run_gw.py:433`)**

The default will silently produce incorrect predictions for season 11 data without a
code change. Derive the current season dynamically from `etl.schema.SEASONS`.

---

**Priority 5 — No DB backup before ETL rebuild (`etl/run.py`)**

A failed ETL mid-rebuild destroys the previous valid database. Adding a `.bak`
rename-before/restore-on-failure guard (see §2.1 above) protects against data loss
in the rare case of an interrupted rebuild.

---

## What Is Already Strong

1. **Temporal CV is done correctly.** Expanding-window folds with no lookahead, baseline
   gate enforcement, and dynamic fold construction when new seasons are added. This is
   the single most important modelling practice for this problem type.

2. **Leakage prevention is systematic.** `shift(1)` before every rolling and lag
   operation, banned features documented at the module level, and imputation fit only on
   the training fold — all with explicit comments explaining why.

3. **Schema versioning for FPL API drift.** `EXPECTED_COLS` in `etl/schema.py` maps
   season_id to expected column frozensets derived from era flags. New columns trigger
   a non-fatal alert rather than silently corrupting the pipeline. This is a
   production-quality approach to external API instability.

4. **Star schema integrity.** Ten post-load validation checks covering orphan keys,
   row counts, season point reconciliation, cost cross-validation, and NULL profiles by
   era. The reconciliation check is correctly scoped to completed seasons.

5. **Monitoring layer with per-GW narrative reports.** Rolling 5-GW MAE with calibrated
   thresholds (1.5× baseline), position-level alerts, and a markdown eval report per GW
   are exactly what you want in a production ML system serving live predictions.

6. **Registry pattern.** New models are added by registering a `ModelSpec` with
   `build_fn` and `predict_fn`. The CV loop, training, and inference are model-agnostic.
   Tier assignment and dependency tracking (for meta-models) are encoded in the spec,
   not scattered across consumers.

7. **Dry-run mode in `retrain_season.py`.** The ability to inspect the full retraining
   execution plan before committing any changes is a thoughtful safety feature.

---

## What Would Make This Production-Ready

The pipeline is already close to production quality. The remaining gaps are:

1. **Automated threshold recalibration at retraining.** `retrain_season.py` step 9
   should compute and suggest new `_THRESHOLDS` values (1.5× new baseline MAE) from the
   fresh CV results, rather than requiring a manual CLAUDE.md edit.

2. **LightGBM early stopping.** Passing `eval_set` to `model.fit()` with an early
   stopping callback eliminates the need to tune `n_estimators` by hand and reduces
   overfitting risk on small position datasets (GK: ~745 training rows in fold 1).

3. **Dynamic season detection.** Replace the hardcoded `season_id=10` default in
   `run_gw.py` with a lookup against `etl.schema.SEASONS`, so the pipeline self-updates
   when a new season row is added.

4. **Environment lock file.** Pin Python version and add a `pyproject.toml` or
   `environment.yml` so the pipeline reproduces exactly on a fresh machine.

5. **Parameterised SQL in dashboard pages.** Replace f-string position embedding with
   `params=` arguments in the two pages that currently use whitelist validation. The
   infrastructure already supports this.

6. **Optuna LightGBM tuning.** As noted in CLAUDE.md, FWD is the highest priority
   position for tuning. Running `python -m ml.train --tune --position FWD` and
   committing the resulting best params to `LGBM_BASE_PARAMS` would close the remaining
   gap between Ridge and LightGBM on that position.

7. **DB backup before ETL rebuild.** The rename-before/restore-on-failure guard
   described in §2.1 protects the most expensive single artefact in the pipeline.
