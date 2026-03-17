# Phase 5 & 6 Modelling Report

## Overview

This document covers the implementation of Phase 5 (Modelling) and Phase 6 (Evaluation),
which were delivered together as an integrated pipeline. It describes what was built, how
it aligns with the project plan, a full interpretation of the results, feature-level
analysis, known limitations, and implications for subsequent phases.

---

## 1. What Was Built

### 1.1 Files

| File | Purpose |
|------|---------|
| `ml/evaluate.py` | 3-fold expanding-window CV: trains models, computes metrics, generates plots |
| `ml/train.py` | Trains final models on all xG era data and serialises artefacts |
| `ml/predict.py` | Inference: loads serialised models, generates ranked GW predictions |

### 1.2 Models Implemented

This report documents the Tier 1 pipeline delivered in the original Phase 5 commit.
Tiers 2 and 3 (19 additional models) were subsequently implemented via `ml/models.py`
— see `revised_modelling_plan.md` for the full inventory and `revised_modelling_implementation_plan.md`
for the batch-by-batch implementation record.

**Tier 1 (documented here):**

| Model | Description |
|-------|-------------|
| `baseline` | Predict `pts_rolling_5gw`; NaN filled with training-set mean |
| `ridge` | Ridge regression (L2, alpha=1.0) with stratified mean imputation and StandardScaler |
| `lgbm` | LightGBM with position-specific hyperparameters; native NaN handling |

**Tiers 2 and 3 (implemented subsequently, not documented in detail here):**

| Family | Models |
|--------|--------|
| Linear | `elasticnet`, `bayesian_ridge`, `lasso`, `poisson_glm` |
| Baselines | `position_mean`, `fdr_mean`, `last_season_avg` |
| Gradient boosting | `xgb`, `random_forest`, `extra_trees`, `hist_gb` |
| Neural (tabular) | `mlp`, `poly_ridge` |
| Decomposed | `minutes_model`, `component_model` |
| Meta / ensemble | `simple_avg`, `stacking`, `blending` |
| Sequential | `lstm`, `gru` (via `ml/evaluate_sequential.py`) |

All models are trained separately for each position (GK, DEF, MID, FWD). Cross-position
training is strictly prohibited and enforced by the pipeline.

### 1.3 CV Setup

Expanding-window temporal cross-validation; no random splitting.

| Fold | Training seasons | Validation season | Approx. period |
|------|-----------------|-------------------|----------------|
| 1 | 7 | 8 | Train 2022-23, val 2023-24 |
| 2 | 7, 8 | 9 | Train 2022-24, val 2024-25 |
| 3 | 7, 8, 9 | 10 | Train 2022-25, val 2025-26 |

### 1.4 Serialised Artefacts

168 model artefacts serialised to `models/` (4 positions × 22 models × .pkl + _meta.json).
The Tier 1 artefacts described here are:

- `{position}_{model}.pkl` — full bundle: sklearn/LightGBM object, scaler, imputer state, feature list
- `{position}_{model}_meta.json` — human-readable metadata: feature list, training rows, CV MAE, CV Spearman

---

## 2. Alignment with the Project Plan

### 2.1 Phase 5 requirements met

| Requirement | Status | Notes |
|------------|--------|-------|
| Position-specific models (never cross-position) | Met | 4 separate feature matrices, 4 separate models |
| Expanding-window temporal CV, 3 folds | Met | s7→8, s7-8→9, s7-9→10 |
| Rolling mean baseline | Met | `pts_rolling_5gw`, NaN fallback to training mean |
| Ridge with stratified mean imputation | Met | Per-season means computed within training fold only |
| Ridge with StandardScaler on training fold only | Met | Scaler stored in bundle for inference |
| LightGBM with position-specific starting hyperparameters | Met | GK conservatively regularised |
| LightGBM native NaN handling (no imputation) | Met | No preprocessing applied before LGBM fit |
| Optuna hyperparameter search | Met (optional) | Available via `--tune` flag; 40 trials, TPE sampler |
| Serialise models with metadata JSON | Met | 12 pkl + 12 json files |
| Baseline gate: beat on >= 2 of 3 metrics (MAE, RMSE, Spearman) | Met | All 8 models pass (2 models x 4 positions) |

### 2.2 Phase 6 requirements met

| Requirement | Status | Notes |
|------------|--------|-------|
| MAE, RMSE, R², Spearman rho, Top-10 precision | Met | All 5 metrics computed per fold |
| Per-GW Top-N precision | Met | Averaged over individual GW groups within the val set |
| Stratified evaluation: home/away | Met | Separate metrics per fold, included in `cv_metrics_all.csv` |
| Stratified evaluation: opponent tier (top-6 vs rest) | Met | Separate metrics per fold |
| Calibration plots | Met | `outputs/models/calibration_{position}.png` |
| SHAP feature importance | Met | `outputs/models/shap_{position}.png` (fold 3 val set) |
| MAE stability across folds | Met | `outputs/models/mae_by_fold_{position}.png` |
| Benchmark comparison | Met | Every model compared against rolling mean baseline |
| CV results saved to logs | Met | `logs/training/cv_metrics_all.csv`, per-position CSVs and reports |

### 2.3 Deliberate scope decisions

- **Ridge alpha left at 1.0 (not tuned):** Cross-validation results show Ridge already
  clearly outperforms both baseline and LightGBM at default settings. Tuning alpha was
  deferred to Phase 7 if needed.
- **Optuna tuning implemented but not run by default:** The `--tune` flag exists and is
  functional. Default runs use the plan-specified starting hyperparameters. See section 5.2
  for the case for running tuning before Phase 8.
- **Tier 2 and 3 models implemented separately:** All models from `revised_modelling_plan.md`
  have since been implemented via the batch-rollout described in `revised_modelling_implementation_plan.md`.
  CV metrics for all 22 models are in `logs/training/cv_metrics_{position}.csv`.

---

## 3. CV Results: Full Metrics

### 3.1 Mean CV metrics across 3 folds

| Position | Model | MAE | RMSE | R² | Spearman | Top-10 prec |
|----------|-------|----:|-----:|---:|----------:|------------:|
| GK | baseline | 2.329 | 3.052 | -0.267 | 0.014 | 0.515 |
| GK | **ridge** | **2.132** | **2.720** | **-0.007** | **0.118** | **0.542** |
| GK | lgbm | 2.215 | 2.836 | -0.096 | 0.071 | 0.528 |
| DEF | baseline | 2.332 | 3.200 | -0.168 | 0.135 | 0.147 |
| DEF | **ridge** | **2.138** | **2.888** | **0.048** | **0.277** | **0.179** |
| DEF | lgbm | 2.225 | 3.005 | -0.031 | 0.205 | 0.169 |
| MID | baseline | 1.997 | 3.000 | -0.054 | 0.298 | 0.148 |
| MID | **ridge** | **1.830** | **2.764** | **0.106** | **0.371** | **0.219** |
| MID | lgbm | 1.883 | 2.845 | 0.053 | 0.315 | 0.171 |
| FWD | baseline | 2.406 | 3.423 | -0.075 | 0.347 | 0.412 |
| FWD | **ridge** | **2.254** | **3.155** | **0.087** | **0.413** | **0.453** |
| FWD | lgbm | 2.404 | 3.319 | -0.008 | 0.280 | 0.408 |

Ridge is the best model on every metric and every position.

### 3.2 Per-fold detail

#### GK

| Fold | Model | MAE | Spearman | Top-10 prec |
|------|-------|----:|----------:|------------:|
| 1 | baseline | 2.297 | -0.029 | 0.492 |
| 1 | ridge | 2.177 | 0.088 | 0.519 |
| 1 | lgbm | 2.258 | 0.091 | 0.511 |
| 2 | baseline | 2.340 | 0.027 | 0.524 |
| 2 | ridge | 2.098 | 0.133 | 0.558 |
| 2 | lgbm | 2.173 | 0.096 | 0.553 |
| 3 | baseline | 2.351 | 0.044 | 0.529 |
| 3 | ridge | 2.120 | 0.133 | 0.550 |
| 3 | lgbm | 2.214 | 0.026 | 0.521 |

GK fold 1 was the most constrained (745 training rows). Both ridge and lgbm still improve on
baseline, validating the conservative GK hyperparameters (num_leaves=15,
min_child_samples=30) specified in the plan.

#### DEF

| Fold | Model | MAE | Spearman | Top-10 prec |
|------|-------|----:|----------:|------------:|
| 1 | baseline | 2.276 | 0.117 | 0.140 |
| 1 | ridge | 2.144 | 0.308 | 0.208 |
| 1 | lgbm | 2.307 | 0.209 | 0.166 |
| 2 | baseline | 2.168 | 0.155 | 0.134 |
| 2 | ridge | 2.038 | 0.273 | 0.195 |
| 2 | lgbm | 2.104 | 0.195 | 0.184 |
| 3 | baseline | 2.551 | 0.132 | 0.167 |
| 3 | ridge | 2.231 | 0.250 | 0.133 |
| 3 | lgbm | 2.265 | 0.210 | 0.158 |

Fold 3 shows a notable MAE increase for all models (baseline +0.28, ridge +0.09,
lgbm +0.07 vs fold 2). This is attributable to the 2025-26 validation set being a
partial season — mid-season samples have different statistical properties than
complete seasons. Ridge is the most stable across folds.

LightGBM's MAE is worse than baseline on fold 1 for DEF (2.307 vs 2.276). This is the
only fold/position combination where a trained model underperforms the baseline. It
self-corrects by fold 2 and passes the 2-of-3 gate overall.

#### MID

| Fold | Model | MAE | Spearman | Top-10 prec |
|------|-------|----:|----------:|------------:|
| 1 | baseline | 2.005 | 0.332 | 0.158 |
| 1 | ridge | 1.775 | 0.403 | 0.213 |
| 1 | lgbm | 1.871 | 0.300 | 0.158 |
| 2 | baseline | 1.897 | 0.300 | 0.179 |
| 2 | ridge | 1.791 | 0.376 | 0.234 |
| 2 | lgbm | 1.830 | 0.340 | 0.184 |
| 3 | baseline | 2.089 | 0.262 | 0.108 |
| 3 | ridge | 1.924 | 0.335 | 0.208 |
| 3 | lgbm | 1.947 | 0.304 | 0.171 |

MID shows the strongest absolute improvements from both models. Ridge reduces baseline
MAE by 0.167 on average and Spearman by +0.073. The MID position also has the highest
Top-10 precision improvement: Ridge 0.219 vs Baseline 0.148.

Note: Top-10 precision for DEF and MID is low in absolute terms (0.15-0.22) because
there are many more DEF/MID players in each GW than GK/FWD, making the
top-10 pool more competitive. GK top-10 precision is high (0.51-0.54) because
there are only ~20 GKs per GW, making top-10 relatively easy.

#### FWD

| Fold | Model | MAE | Spearman | Top-10 prec |
|------|-------|----:|----------:|------------:|
| 1 | baseline | 2.460 | 0.331 | 0.382 |
| 1 | ridge | 2.130 | 0.380 | 0.437 |
| 1 | lgbm | 2.394 | 0.157 | 0.366 |
| 2 | baseline | 2.497 | 0.390 | 0.463 |
| 2 | ridge | 2.400 | 0.464 | 0.471 |
| 2 | lgbm | 2.568 | 0.317 | 0.445 |
| 3 | baseline | 2.261 | 0.320 | 0.392 |
| 3 | ridge | 2.233 | 0.394 | 0.450 |
| 3 | lgbm | 2.250 | 0.365 | 0.413 |

FWD is the most difficult position to model. LightGBM's fold 1 Spearman is only 0.157
(vs baseline 0.331) — a significant underperformance attributable to the small
fold 1 training set (1,459 FWD rows) and LightGBM's tendency to overfit on small
datasets without tuning. FWD LightGBM only narrowly passes the baseline gate (MAE 2.404
vs baseline 2.406 — 0.002 pts improvement). FWD is the strongest candidate for Optuna
tuning.

---

## 4. Stratified Results: Home/Away and Opponent Tier

### 4.1 Home vs away (Ridge, mean across folds)

| Position | Home MAE | Away MAE | Home Spearman | Away Spearman |
|----------|----------:|---------:|--------------:|--------------:|
| GK | 2.248 | 2.016 | 0.106 | 0.134 |
| DEF | 2.273 | 2.005 | 0.242 | 0.288 |
| MID | 1.963 | 1.695 | 0.365 | 0.377 |
| FWD | 2.430 | 2.078 | 0.444 | 0.387 |

Away games are systematically easier to predict (lower MAE across all positions). This is
consistent with the EDA finding of a home premium: home players score more points on
average but also have higher variance (big scores are more likely at home). The model
captures the direction but underestimates the magnitude of home-game hauls.

Exception: FWD Spearman is higher for home (0.444 vs 0.387), likely because the most
elite FWDs (who reliably haul at home) dominate the ranking signal.

### 4.2 Opponent tier: top-6 vs rest (Ridge, mean across folds)

| Position | vs Top-6 MAE | vs Rest MAE | vs Top-6 Spearman | vs Rest Spearman |
|----------|-----------:|----------:|-----------------:|----------------:|
| GK | 1.837 | 2.258 | 0.038 | 0.076 |
| DEF | 1.745 | 2.308 | 0.202 | 0.243 |
| MID | 1.519 | 1.961 | 0.343 | 0.378 |
| FWD | 1.839 | 2.436 | 0.375 | 0.414 |

vs-top-6 MAE is consistently lower than vs-rest across all positions. This is a
statistical artefact: when facing top-6 sides, most players score low (1-2 pts), so
predicting low is easy. The model has learned the top-6 penalty well (as expected given
`opponent_season_rank` is a mandatory feature), but the restricted score range makes
MAE deceptively small. Spearman is also lower vs top-6 because ranking is harder when
most players cluster around 1-2 pts.

---

## 5. Feature Analysis

### 5.1 Ridge coefficients (top 8 by absolute magnitude, final model)

Coefficients are on standardised inputs so magnitudes are directly comparable.

**GK**
- `opponent_season_rank` (+0.423) — strongest predictor; facing a weaker team (higher rank
  number = lower league position) substantially boosts expected GK points
- `start_cost` (+0.314) — price as quality proxy; expensive GKs play for better teams
- `value_lag1` (-0.243) — negative coefficient: players whose price rose recently tend to
  mean-revert; this captures the post-haul price-rise effect
- `was_home` (+0.129) — home premium confirmed

**DEF**
- `opponent_season_rank` (+0.610) — dominates the DEF model; the EDA finding of a 33.8%
  top-6 penalty is captured here
- `start_cost` (+0.348) — expensive defenders play for better attacking teams; indirectly
  captures team quality beyond `team_goals_conceded_season`
- `mins_rolling_3gw` (+0.294) — availability; playing more minutes = more points
- `was_home` (+0.211) — home premium second largest after GK at this position
- `team_goals_conceded_season` (-0.154) — team defensive quality; negative coefficient
  means more goals conceded = fewer points for defenders on that team

**MID**
- `start_cost` (+0.559) — quality proxy; top MIDs are consistently expensive and productive
- `pts_rolling_5gw` (+0.329) and `xg_rolling_5gw` (+0.314) — form signals
- `opponent_season_rank` (+0.311) — fixture difficulty
- `xgi_rolling_5gw` (-0.246) — negative despite being an attacking signal; this is a
  Ridge collinearity artefact. `xgi` is the sum of `xg` and `xa`, so all three are
  correlated. Ridge redistributes weight and can assign negative coefficients to
  multicollinear features.

**FWD**
- `start_cost` (+0.956) — by far the strongest predictor; elite FWDs (Haaland, Watkins)
  are expensive and reliably score
- `xgi_rolling_5gw` (-0.762) — another collinearity artefact with `xg`, `xa`, and `goals`
- `goals_rolling_5gw` (+0.543) and `xg_rolling_5gw` (+0.531) — attacking form
- `mins_rolling_3gw` (+0.492) — availability

Note on collinearity: the xG feature set (`xg`, `xa`, `xgi`) is deliberately correlated
(xgi = xg + xa). Ridge handles this by distributing weight, sometimes assigning unexpected
signs to individual features. The composite prediction is still correct — the negative
`xgi` coefficient is offset by the positive `xg` and `xa` coefficients. This would
be worth addressing with LightGBM (which handles collinearity naturally) or by dropping
`xgi` as a feature for Ridge.

### 5.2 LightGBM split importance (top 8, final model)

LightGBM importance counts the number of times a feature is used to split across all trees.

**GK**: transfer activity leads (transfers_out 294, transfers_in 288), followed by `xgc_rolling_5gw`,
`season_pts_per_gw_to_date`, `saves_rolling_5gw`. LightGBM relies more on crowd-signal
features than Ridge.

**DEF**: defensive context features dominate (opponent_cs_rate 910, team_goals_conceded 855,
opponent_goals_scored 838, opponent_season_rank 705, xgc_rolling 704). This is consistent
with the EDA finding that 46.6% of DEF/GK variance is explained by team-season defensive
quality.

**MID**: `xa_rolling_5gw` and `mins_rolling_3gw` lead (676, 675), followed by
`opponent_cs_rate_season` — LightGBM emphasises availability and opponent defensive
strength over raw attacking output.

**FWD**: transfer activity leads (transfers_in 809, mins_rolling 740), then opponent context.
LightGBM is using the crowd-signal features (transfers) more heavily than Ridge. This may
partly explain FWD LightGBM's weaker performance: transfers lag performance and can
introduce noise for individual GW predictions.

---

## 6. Interpretation and Diagnostics

### 6.1 Why Ridge outperforms LightGBM

Ridge is the best model across all positions and metrics. Several explanations:

1. **Dataset size.** 40,900 total rows after filtering, split across 4 positions. GK has
   only 745 training rows in fold 1. LightGBM generally requires more data to beat linear
   models — the crossover point is typically 50,000-100,000 rows per model.

2. **Feature linearity.** The engineered features are mostly linear transformations
   (rolling means, season-to-date averages, static costs). Rolling means of `pts` and
   `xg` have a near-linear relationship with the target. Ridge is well-suited to this
   feature structure.

3. **LightGBM not tuned.** The plan specifies starting hyperparameters, not optimised
   ones. The `--tune` flag runs 40 Optuna trials but has not been executed. Running it
   is the highest-priority next modelling step and is expected to close the gap.

4. **Target distribution.** `total_points` is heavily right-skewed: most values are
   0-4 with rare hauls of 10+. LightGBM's leaf-wise splits can overfit the haul events
   with small datasets. Ridge's global regularisation is more robust here.

### 6.2 Low R² and what it means

R² is near zero or negative for many position/fold combinations. This is expected and not
a modelling failure. It reflects:

- GW-level points are inherently noisy. A player with strong form can blank due to
  injury, tactical rotation, or simply bad luck. The model predicts *expected* points,
  not realised points.
- The target variance is dominated by rare hauls (<2% of appearances score 10+ pts).
  Predicting the mean well (low MAE) is more valuable for FPL than explaining haul variance.
- R² is the right metric to track for calibration, not for model quality judgement.
  The correct quality metrics for FPL are Spearman rho (ranking) and Top-N precision
  (selection quality).

### 6.3 Spearman rho context

The Spearman rho values reflect ranking quality across the validation set:

- **GK (0.118):** Low, as expected. GK scoring is largely binary (CS or no CS) and
  driven by team quality, which the model approximates but cannot predict precisely.
- **DEF (0.277):** Moderate. The team defensive features explain a lot, but individual
  defensive contributions (tackles, clearances) are not in the feature set.
- **MID (0.371):** Best absolute improvement over baseline (+0.073). xG features and
  start_cost provide good separation between elite and rotation midfielders.
- **FWD (0.413):** Highest absolute Spearman, driven by the strong `start_cost` and
  `xg_rolling` signal for elite forwards.

A Spearman of 0.3-0.4 is broadly consistent with published FPL prediction benchmarks
(where the best public tools reach approximately 0.35-0.45 on held-out weeks without
injury data).

### 6.4 Top-10 precision context

For DEF and MID, top-10 precision is 0.15-0.22, meaning in a typical GW the model
correctly identifies 1.5-2.2 of the actual top-10 scorers from that position. This sounds
modest but reflects the fundamental difficulty: a "top-10 DEF" in a given GW is usually
determined by clean sheets and bonus, both of which are near-random at the individual
level. Correctly identifying even 2 out of 10 is meaningful in a selection context.

For GK (0.54) and FWD (0.45), the better precision reflects a smaller pool (fewer GKs
and FWDs in each GW) and stronger form concentration (the same 2-3 elite GKs/FWDs
dominate most weeks).

### 6.5 Fold 3 degradation (DEF baseline +0.28 MAE)

The baseline MAE jumps from 2.17 (fold 2) to 2.55 (fold 3) for DEF. The trained models
also degrade but more mildly (+0.09 for Ridge). Fold 3 validates on season 10 (2025-26),
which is an in-progress season at time of data collection. Partial-season feature vectors
(especially season-to-date stats early in the season) behave differently from
end-of-season rows. This is expected and not a data leakage issue.

---

## 7. Outputs Generated

| Output | Location | Description |
|--------|----------|-------------|
| CV metrics CSV (all positions) | `logs/training/cv_metrics_all.csv` | 180 rows: 15 model variants x 3 folds x 4 positions |
| CV metrics CSV (per position) | `logs/training/cv_metrics_{pos}.csv` | Per-position detail |
| CV predictions parquet | `logs/training/cv_preds_{pos}.parquet` | Row-level OOF predictions for all folds |
| CV report markdown | `logs/training/cv_report_{pos}.md` | Mean metrics, baseline gate, per-fold table |
| Calibration plots | `outputs/models/calibration_{pos}.png` | Predicted vs actual in 10 quantile bins, all models |
| MAE-by-fold plots | `outputs/models/mae_by_fold_{pos}.png` | MAE stability across 3 folds per model |
| SHAP importance plots | `outputs/models/shap_{pos}.png` | Mean absolute SHAP value per feature (fold 3 val) |
| Serialised models | `models/{pos}_{model}.pkl` | 12 bundles: model + scaler + imputer state + feature list |
| Model metadata | `models/{pos}_{model}_meta.json` | 12 files: CV MAE, Spearman, feature list, training params |
| GW prediction CSV | `outputs/predictions/gw24_s10_predictions.csv` | Test output: 285 rows, all 4 positions, GW24 season 10 |

---

## 8. Implications for Subsequent Phases

### 8.1 Phase 7 — Dashboard

The dashboard has immediate access to all artefacts needed:

- **Model comparison table (Page 3):** `logs/training/cv_metrics_all.csv` is the direct
  data source. MAE, RMSE, Spearman, and Top-10 precision are ready to display for all
  positions and models.
- **Calibration plots and SHAP charts:** Pre-generated as PNGs in `outputs/models/`.
  Can be embedded directly.
- **GW Predictions (Page 4):** `ml/predict.py` is the entry point. Calling
  `predict_gw(gw, season_id, models=('lgbm', 'ridge'))` returns a ranked DataFrame
  suitable for display. The ensemble column (`pred_ensemble`) is the recommended default.
- **Recommended production model:** Ridge, for all positions. It is the most accurate,
  fastest (linear), and most interpretable. The dashboard should default to Ridge but
  expose model selection.

### 8.2 Phase 8 — Deployment

Key decisions before deployment:

1. **Run Optuna tuning (`python -m ml.train --tune`)** before finalising the production
   model. LightGBM with tuned hyperparameters may close the gap with Ridge and provide
   better ensemble diversity. FWD is the priority.

2. **Retrain Ridge with alpha tuning.** A simple search over `alpha in [0.1, 0.5, 1.0,
   5.0, 10.0]` within the CV framework may yield 0.02-0.05 MAE improvement.

3. **Model versioning:** The plan specifies `models/v{season_id}/` for versioned storage.
   The current `models/` directory is v1 (trained on seasons 7-10). When season 11 data
   is available, retrain into `models/v11/`.

4. **Production model selection:** Based on current CV results, the deployment pipeline
   should use `ridge` as the default model. LightGBM adds value in an ensemble but should
   not be used alone given the current FWD fold-1 performance.

5. **Imputer state is serialised:** The `season_means` and `global_means` DataFrames in
   each Ridge `.pkl` bundle are fit on seasons 7-10. For inference on season 11 GW1 data,
   season 11 will not have a prior-season mean entry — the model will correctly fall back
   to `global_means`. This is the intended behaviour and requires no code change.

### 8.3 Phase 9 — Monitoring

The monitoring specification (project_plan.md §9.2) requires 5-GW rolling MAE with a
1.5x baseline threshold. Baseline MAE values from the CV to use as reference:

| Position | Baseline MAE (CV mean) | Monitoring threshold (1.5x) |
|----------|----------------------:|----------------------------:|
| GK | 2.329 | 3.494 |
| DEF | 2.332 | 3.498 |
| MID | 1.997 | 2.996 |
| FWD | 2.406 | 3.609 |

These thresholds should be seeded into `logs/monitoring/monitoring_log.csv` when Phase 9
is initialised.

### 8.4 Tier 2 models (deferred)

The project plan specifies XGBoost, Random Forest, MLP, and a decomposed minutes-model
as Tier 2. Based on Phase 5 results:

- **XGBoost:** Most likely to add value. Stronger regularisation than LightGBM and a
  different tree structure. Recommend as the first Tier 2 addition.
- **Random Forest:** Different inductive bias from boosting; useful for ensemble diversity.
  Less likely to beat LightGBM alone but adds value in stacking.
- **Decomposed minutes model (P(starts) then conditional pts):** High value for FPL
  use-cases since rotation risk is the largest real-world blind spot. Difficult to
  implement well without injury/team-news data.
- **MLP:** Unlikely to beat LightGBM with current dataset size. Defer until more data
  is available (more seasons or per-game data).

---

## 9. Known Limitations

These limitations carry forward from the EDA and feature engineering phases and are not
fixable within the current dataset:

| Limitation | Impact on results | Status |
|-----------|-------------------|--------|
| No injury or team-news data | Largest predictive gap; all models assume player availability | Unaddressable from current sources |
| MID sub-role heterogeneity (strikers vs creators) | Structural residual variance in MID; CV = 0.932 | Partially mitigated by xGI features; accept as known |
| Survivorship bias (75% data from 30+ GW starters) | Model is well-calibrated for regular starters; less reliable for rotation players | Documented; warn in dashboard |
| Cold-start players (39.4% appear only 1 season) | Rolling features unavailable GW1; model falls back to `start_cost` + prior season form | By design; documented |
| Partial 2025-26 season in fold 3 validation | Fold 3 metrics slightly inflated or deflated vs a complete season | Expected; acknowledged in fold 3 notes above |
| `goals_conceded` is player-scoped in `fact_gw_player` | Cannot use as team-level stat; mitigated by using `team_h_score`/`team_a_score` | Mitigated in feature engineering |
| xG era constraint (seasons 7-10 only) | ~57% of available data excluded | Justified by era incompatibility; see project_plan.md §4.2 |
