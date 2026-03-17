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
| 5 | Modelling |  Complete |
| 6 | Evaluation |  Complete |
| 7 | Interactive Dashboard & Visualisations |  Not started |
| 8 | Deployment |  Complete |
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

**Option A — xG era only (seasons 7–10, 2022-23 to 2025-26).**

Post-filter row counts (after base filter): GK 2,731 | DEF 13,723 | MID 19,495 | FWD 4,951 | total 40,900.
The ~96,000 figure was pre-filter; the base filter removes ~57% of rows, primarily through the
`minutes > 0` condition and the in-progress 2025-26 season.

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
- `xgi_rolling_5gw` — combined attacking threat (note: collinear with xg + xa since xgi = xg + xa;
  Ridge assigns negative coefficients to xgi for MID/FWD as a collinearity artefact — composite
  prediction is still correct, but dropping xgi from Ridge for MID/FWD is recommended)
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

> **Note:** `docs/modelling_plan.md` consolidates the full model inventory, tier assignments,
> priority-ordered implementation roadmap, registry architecture, batch sequencing, per-model
> bundle specs, and verification gate outcomes. `docs/modelling_evaluation_report.md` contains
> all CV results, stratified analyses, and phase implications. The sections below record the
> core training setup decisions made during implementation.

**Status: Complete.**

**Deliverables:**
- `ml/models.py` — central model registry (ModelSpec dataclass, build_fn / predict_fn for all 22 models)
- `ml/evaluate.py` — 3-fold expanding-window CV; Pass 1 (tabular/decomposed), Pass 2 (meta); metrics, calibration plots, SHAP plots
- `ml/evaluate_sequential.py` — standalone CV pipeline for LSTM / GRU (sequence reshaping, separate from tabular loop)
- `ml/evaluate_phase6.py` — post-hoc Phase 6 evaluation: minutes bucket, price band, residual plots, learning curves
- `ml/train.py` — final model training on all xG era data; `--meta` flag for OOF-based meta-model training
- `ml/predict.py` — inference pipeline; auto-chains base models for meta-model predictions; per-GW ranked output
- `models/` — 168 serialised artefacts (4 positions × 21 serialised models × .pkl + _meta.json)
- `logs/training/` — CV metrics CSVs, OOF predictions parquets, per-position markdown reports, sequential CV metrics
- `outputs/models/` — calibration, MAE-by-fold, SHAP, residuals (per position) and learning curves plots (17 PNGs total)
- `docs/modelling_evaluation_report.md` — full modelling and evaluation report: all CV results, all stratifications, residual analysis, learning curves, Phase 7+ implications
- `docs/modelling_plan.md` — model inventory, tier rationale, implementation strategy, batch specs, and verification gate outcomes

**Key results:**
- **Production model:** Ridge for all positions. CV MAE: GK 2.132 | DEF 2.138 | MID 1.830 | FWD 2.254
- **Best ensemble:** Blending (ridge + bayesian_ridge + poisson_glm + mlp); beats Ridge on GK and DEF
- **Uncertainty:** BayesianRidge pred_std available per prediction for dashboard confidence bands
- **Sequential:** LSTM and GRU beat LightGBM on 3/4 positions but do not beat Ridge; not serialised
- **Baseline gate:** 17 of 20 models pass across all 4 positions; 3 failures (fdr_mean, lasso, poly_ridge) are documented expected outcomes
- **Monitoring thresholds (1.5× baseline MAE):** GK 3.494 | DEF 3.498 | MID 2.996 | FWD 3.609

---

**Target:** `total_points` (GW-level regression, continuous output)
**Architecture:** Four position-specific models — GK, DEF, MID, FWD. Never cross-position.
**Validation:** Expanding-window temporal CV (train on seasons N…k, test on k+1). Never random split.

The full model brainstorm, tier assignments, priority rationale, and batch-by-batch implementation
record are in `docs/modelling_plan.md`. The CV results, feature analysis, and all phase
implications are in `docs/modelling_evaluation_report.md`. The sections below record the core training
setup decisions made during implementation.

---

### 5.1 Training Setup

**File structure:**
```
ml/
├── features.py              # build_feature_matrix(position, era)
├── models.py                # central registry: ModelSpec, all build_fn/predict_fn
├── train.py                 # train and serialise models [--meta]
├── evaluate.py              # CV evaluation + metrics + calibration/SHAP plots
├── evaluate_sequential.py   # LSTM/GRU CV (separate pipeline)
├── evaluate_phase6.py       # post-hoc: minutes bucket, price band, residuals, learning curves
└── predict.py               # inference on new GW fixture list
models/
└── {position}_{model}.pkl
```

**NaN handling:** Feature matrices contain ~4.5% NaN in player rolling features (first GW
of each player-season) and ~2.8% NaN in team/opponent features (first fixture of each
team-season). LightGBM handles NaN natively — no action needed. Ridge requires imputation
before `fit()`; use mean imputation computed within the training fold only, stratified by
`season_id`. Global fallback means are also stored in the bundle for inference on unseen
seasons. Never fit the imputer on the full dataset or on the validation fold.

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

**Meta-model training:** Meta-models (simple_avg, stacking, blending) cannot be trained on a
single full-data pass — they require out-of-fold base model predictions. Run `ml/evaluate.py`
first to generate `logs/training/cv_preds_{pos}.parquet`, then use `ml/train.py --meta` to
fit meta-learners on the full 3-fold OOF stack.

**Reproducibility:** Set `random_state=42` everywhere; log all hyperparameters to
`logs/training/`.

---

## Phase 6 — Evaluation

**Status: Complete.**
See `docs/modelling_evaluation_report.md` for full CV results, all stratifications, residual
analysis, learning curves, and the Phase 6 completion checklist.

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

**Metric interpretation (confirmed by Phase 5/6 CV):**
- **R² near zero or negative is expected** — GW-level FPL points are inherently noisy; a player
  can blank due to rotation or bad luck regardless of predicted form. Predicting the mean well
  (low MAE) is more valuable than explaining haul variance. Positive R² was achieved on DEF
  (0.048), MID (0.106), and FWD (0.087) with Ridge.
- **Spearman ρ of 0.3–0.4 is competitive** — consistent with published FPL prediction benchmarks
  where the best public tools reach approximately 0.35–0.45 on held-out GWs without injury data.
  GK (0.118) is structurally lower because GK scoring is largely binary (CS or no CS).
- **Top-10 precision of 0.15–0.22 for DEF/MID is meaningful** — a "top-10 DEF" in a given GW
  is usually determined by clean sheets and bonus, which are near-random at the individual level.
  GK (0.54) and FWD (0.45) show higher precision because of smaller pools and stronger form
  concentration among elite players.

### 6.3 Stratified evaluation
Metrics computed separately for:
- **Position (GK / DEF / MID / FWD)** — all models trained and evaluated position-specifically.
- **Home vs away** — implemented. Away games are systematically easier to predict (lower MAE across
  all positions). Home-game hauls have higher variance that the model underestimates. Exception:
  FWD Spearman is higher for home games (0.444 vs 0.387), as elite FWDs dominate home rankings.
  Ridge home vs away MAE: GK 2.248 / 2.016, DEF 2.273 / 2.005, MID 1.963 / 1.695, FWD 2.430 / 2.078.
- **Opponent tier: top-6 vs rest** — implemented. vs-top-6 MAE is consistently lower than vs-rest
  (Ridge GK: 1.837 vs 2.258, DEF: 1.745 vs 2.308, MID: 1.519 vs 1.961, FWD: 1.839 vs 2.436).
  This is a statistical artefact: top-6 fixtures compress scores to 1–2 pts, making low predictions
  easy to validate. Spearman is also lower vs top-6 because ranking is harder when players cluster
  at the same score.
- **Minutes bucket** — implemented (`ml/evaluate_phase6.py`). Cameos (<30 min) have lowest absolute
  MAE but near-zero Spearman; starters (60+) have highest MAE and the most useful rank signal.
- **Price band** — implemented (`ml/evaluate_phase6.py`). MAE increases with price; Spearman
  declines with price (Budget > Mid > Premium > Elite across DEF, MID, FWD).

### 6.4 Calibration & diagnostics
- **Calibration plots** — implemented; `outputs/models/calibration_{pos}.png`
- **MAE-by-fold plots** — implemented; `outputs/models/mae_by_fold_{pos}.png`
- **SHAP summary plots** — implemented; `outputs/models/shap_{pos}.png` (LightGBM, fold 3)
- **Residual analysis** — implemented (`ml/evaluate_phase6.py`); `outputs/models/residuals_{pos}.png`.
  Consistent mild positive heteroscedasticity (ρ = 0.33–0.55) across all positions. Model
  over-predicts against top-5 opponents across all positions.
- **Learning curves** — implemented (`ml/evaluate_phase6.py`); `outputs/models/learning_curves.png`.
  Ridge–LightGBM gap narrows from fold 1 to fold 3 on DEF/MID/FWD, suggesting Optuna tuning
  will close it further.

### 6.5 Benchmark comparison table
Every model evaluated against the rolling-mean baseline. A model must beat the baseline on
at least 2 of 3 primary metrics (MAE, RMSE, Spearman ρ) to be considered for production.
Full results are in `docs/modelling_evaluation_report.md`. Summary of key results:

| Position | Baseline MAE | Ridge MAE | Best model MAE | Best model |
|----------|------------:|----------:|---------------:|------------|
| GK | 2.329 | 2.132 | 2.098 | minutes_model |
| DEF | 2.332 | 2.138 | 1.994 | component_model |
| MID | 1.997 | 1.830 | 1.822 | poisson_glm |
| FWD | 2.406 | 2.254 | 2.155 | component_model |

**Gate pass rate:** 17 of 20 models pass across all 4 positions. The 3 consistent failures
(fdr_mean, lasso, poly_ridge) have documented structural limitations. All 8 original Tier 1
models (Ridge and LightGBM, all 4 positions) pass.

### 6.6 Known limitations to document

| Limitation | Quantified impact | Mitigation status |
|------------|:-----------------:|-------------------|
| No injury or team-news data | Largest predictive gap; all models assume player availability | Unaddressable from current sources; document in dashboard |
| MID sub-role heterogeneity | CV R² = 0.106 (positive but low); structural residual from striker-vs-creator mixing | Partially mitigated by xGI rolling features; residual is structural |
| Survivorship bias | 75% of data from 30+ GW starters | Less reliable for rotation/fringe players; warn users in dashboard |
| Cold-start players (no prior-season history) | 39.4% of players appear only 1 season; rolling features unavailable at GW1 | Fall back to `start_cost` + position priors; `last_season_avg` handles GW1 specifically |
| Team quality confounding (DEF/GK) | 46.6% variance in DEF/GK pts explained by team goals conceded alone | Mitigated by `team_goals_conceded_season`; residual individual-skill signal remains weak |
| Partial 2025-26 season in fold 3 | Fold 3 DEF baseline MAE +0.28 vs fold 2; Ridge degrades +0.09 | Expected; partial-season feature vectors behave differently from end-of-season rows |
| xG era constraint (seasons 7–10 only) | ~57% of available rows excluded by base filter; 40,900 rows remain | Justified by era incompatibility; Option B deferred — see §4.2 |

---

## Phase 7 — Interactive Dashboard & Visualisations

### 7.1 Architecture
- **Framework:** Streamlit (lower barrier; faster to iterate) or Plotly Dash (more
  control for production). Decide at Phase 7 kickoff.
- **Static charts:** matplotlib / seaborn exported to `outputs/eda/`
- **Serving:** local (`streamlit run app.py`); no cloud deployment in initial phase
- **Output directory:** `outputs/dashboards/`

### 7.2 Model selection for dashboard

All 21 serialised models are available via `ml/predict.py`. The following are
recommended for the dashboard based on Phase 5/6 CV results:

| Role | Model | Rationale |
|------|-------|-----------|
| Default production | `ridge` | Best MAE and Spearman on all positions at default settings; sub-second inference; fully interpretable |
| Uncertainty bands | `bayesian_ridge` | Produces `pred_std` per prediction; near-identical point estimates to Ridge; enables "predicted pts ± uncertainty" display |
| Ensemble view | `blending` | Best ensemble model; beats Ridge on GK (2.123 vs 2.132) and DEF (2.121 vs 2.138); deps: ridge, bayesian_ridge, poisson_glm, mlp |
| Component scouting | `component_model` | Best MAE on DEF (1.994) and FWD (2.155); sub-predictions (goals, assists, CS, bonus) surfaceable for player breakdown UI |
| Rotation risk | `minutes_model` | Produces P(starts) intermediate; best GK MAE (2.098); P(starts) column is a direct rotation-risk signal |

Dashboard should default to Ridge but expose model selection. The model selector can offer at
minimum: Ridge, Bayesian Ridge, Blending, and an optional "all models" comparison view.

### 7.3 Dashboard pages / sections

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
- **Data source:** `logs/training/cv_metrics_{pos}.csv` (21 models × 3 folds × 5 metrics, all positions)
- Model comparison table: MAE / RMSE / Spearman ρ / Top-10 precision per model × position
- **Calibration plots:** pre-rendered at `outputs/models/calibration_{pos}.png` — embed directly
- **MAE-by-fold plots:** pre-rendered at `outputs/models/mae_by_fold_{pos}.png` — stability view
- **SHAP feature importance:** pre-rendered at `outputs/models/shap_{pos}.png` (LightGBM, fold 3)
- Residual plot: OOF residuals from `logs/training/cv_preds_{pos}.parquet` vs GW, coloured by position

#### Page 4 — GW Predictions
- **Data source:** `ml/predict.py` → `predict_gw(gw, season_id, models=(...))` returns ranked DataFrame
- Model selector (default: ridge; options: bayesian_ridge, blending, ridge, and others)
- Ranked player predictions table: player, position, team, opponent, predicted pts
- **Uncertainty column:** when bayesian_ridge selected, show "predicted pts ± pred_std"
- **Rotation risk column:** when minutes_model selected or alongside ridge, show P(starts) %
- **Component breakdown:** when component_model selected, show goals/assists/CS/bonus sub-predictions
- Filter by position, price band, top-N per position
- Download as CSV button; output path: `outputs/predictions/gw{N}_s{season}_predictions.csv`

#### Page 5 — Player Scouting (optional extension)
- Component model sub-predictions: expected goals, assists, CS probability, bonus per player
- Rotation risk table: P(starts) from minutes_model, sorted by risk level
- Price vs predicted pts scatter per position (value pick identification)

### 7.4 Static report charts (always export)
- `outputs/eda/points_distribution.png`
- `outputs/eda/home_away_effect.png`
- `outputs/eda/team_strength_heatmap.png`
- `outputs/models/calibration_{position}.png`
- `outputs/models/mae_by_fold_{position}.png`
- `outputs/models/shap_{position}.png`

### 7.5 Sequential model integration (deferred)

LSTM and GRU are not serialised and cannot be called from `ml/predict.py`. If sequential
predictions are desired in the dashboard, run `ml/evaluate_sequential.py` once per GW to
generate a prediction CSV, then load it alongside the tabular predictions. A full
`torch.save` serialisation path can be added to `evaluate_sequential.py` if justified.

---

## Phase 8 — Deployment (Complete)

**Delivered:** branch `feature/phase8-deployment`, commits `97851e2`–`b2408a8`.
**Report:** `docs/deployment_report.md`

### 8.1 Pre-deployment model improvements (delivered)

**Ridge alpha search** (`--alpha-search` flag on `ml/train.py`):
Best alphas — GK: 10.0 | DEF: 0.1 | MID: 0.1 | FWD: 0.1
Results logged to `logs/training/ridge_alpha_search.csv`.

**Drop `xgi_rolling_5gw` from Ridge for MID and FWD** (`ml/models.py _build_ridge`):
Eliminates negative xgi coefficient caused by xg+xa+xgi collinearity. MID/FWD lose 0.005/0.016
MAE compared to Phase 6 (expected cost of removing a redundant but correlated feature).

**Updated CV mean MAE (Ridge, post-Phase-8):**
GK 2.130 (alpha=10.0) | DEF 2.138 (alpha=0.1) | MID 1.835 (alpha=0.1) | FWD 2.270 (alpha=0.1)

All 168 model artefacts and meta-models retrained.

**Optuna LightGBM tuning:** deferred. `python -m ml.train --tune` is implemented and ready.
FWD is highest priority. Run before end-of-season retraining.

### 8.2 Model serialisation (delivered)
- All 168 artefacts serialised: `models/{position}_{model}.pkl` + `models/{position}_{model}_meta.json`
- `.pkl` files are gitignored (binary). `_meta.json` files are committed.
- **Default production model:** `ridge` for all positions.
- Versioned model directories (`models/v{season_id}/`) are not yet implemented — use flat layout for now.

### 8.3 Live data pipeline (delivered)

**`etl/fetch.py`** — FPL API client:
- `fetch_bootstrap()`, `fetch_fixtures()`, `fetch_gw_live(gw)` — three primary endpoints
- `build_merged_gw()` — transforms API response to `merged_gw.csv` schema
- `write_season_csvs()` — appends GW rows (deduplicates on GW+fixture); overwrites `players_raw.csv`
- Exponential backoff retry (3 attempts); `FetchError` on critical endpoint failure
- Standalone: `python -m etl.fetch --gw 30 --season 2025-26`

**`run_gw.py`** — end-to-end GW runner:
```
Step 1  Fetch    etl/fetch.py writes CSVs to data/{season}/
Step 2  ETL      python -m etl.run (full rebuild, ~16s, 11 validation checks)
Step 3  Predict  ml/predict.predict_gw() -> outputs/predictions/gw{N}_s{season}_predictions.csv
Step 4  Monitor  MAE/RMSE/Spearman/top-10 -> logs/monitoring/monitoring_log.csv
```
Flags: `--gw`, `--season`, `--skip-fetch`, `--skip-etl`, `--model`

**ETL validation fix:** Points reconciliation check (#6 in `etl/validate.py`) now scoped to
completed seasons only. The in-progress season is excluded — FPL retroactive corrections
accumulate in `players_raw.csv` but are not back-propagated to individual GW rows.

**Feature matrix cache:** `outputs/features/*.parquet` must be cleared after adding new GWs.
Cache is rebuilt automatically on next `build_feature_matrix` call.

**Tested:** GW 24 (static data) and GW 30 (live fetch from FPL API). All 11 ETL checks pass.
No monitoring alerts on either GW.

### 8.4 Monitoring infrastructure (delivered)

`logs/monitoring/monitoring_log.csv` — schema:
`season_id, gw, model, position, mae, rmse, spearman, top10_precision, rolling_mae_5gw, threshold, alert, logged_at`

Alert thresholds (1.5× baseline MAE): GK 3.494 | DEF 3.498 | MID 2.996 | FWD 3.609

### 8.5 Retraining cadence
- **End of season:** `python -m ml.train --all` with new season appended. Run full CV on new fold.
- **Mid-season trigger:** 5-GW rolling MAE > threshold in monitoring log → flag for review.

---

## Phase 9 — Monitoring

### 9.1 Per-GW performance tracking
After each GW result is published:
1. Join `gw{N}_s{season}_predictions.csv` against actual `fact_gw_player` results for that GW
2. Compute: MAE, RMSE, Spearman ρ, top-10 precision for that GW
3. Append to `logs/monitoring/monitoring_log.csv`

### 9.2 Rolling metrics and alert thresholds

Compute 5-GW rolling MAE per position per model. Thresholds are 1.5× the CV baseline MAE
(rolling-mean baseline, averaged across 3 folds), seeded from Phase 5/6 results:

| Position | Baseline MAE (CV mean) | Alert threshold (1.5×) |
|----------|----------------------:|----------------------:|
| GK | 2.329 | 3.494 |
| DEF | 2.332 | 3.498 |
| MID | 1.997 | 2.996 |
| FWD | 2.406 | 3.609 |

Seed these values into `logs/monitoring/monitoring_log.csv` at Phase 9 initialisation.
Flag and review if any model's 5-GW rolling MAE exceeds the threshold for its position.
Trigger retraining if confirmed performance degradation (not a one-GW spike).

### 9.3 Schema change alerting
FPL has added new column groups each season (xG in 2022-23, manager mode in 2024-25,
defensive stats in 2025-26). Before each season's data is loaded:
- Check `merged_gw.csv` column set against `etl/schema.py` era definitions
- Alert if new columns appear or existing columns are dropped
- Update `etl/schema.py`, `ml/features.py`, and `project_plan.md` accordingly

### 9.4 Output
- `logs/monitoring/monitoring_log.csv` — per-GW metrics with alert threshold columns
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
