# Revised Modelling Implementation Plan

This document defines the implementation strategy for all models listed in
`revised_modelling_plan.md`. Tier 1 (rolling-mean baseline, Ridge, LightGBM) is
complete. This plan covers the remaining 22 models across Tiers 1–3, plus the
architecture refactor required to support them cleanly.

---

## Architecture Decision: Hybrid Registry + Family Adapter

A new `ml/models.py` file acts as the single source of truth for all model definitions.
It maps each model name to a `ModelSpec` dataclass:

```python
@dataclass
class ModelSpec:
    name: str
    family: str          # 'tabular' | 'meta' | 'sequential' | 'decomposed'
    tier: int
    requires_imputation: bool   # call stratified_impute inside build_fn
    requires_scaling: bool      # fit StandardScaler inside build_fn
    build_fn: Callable          # (X_train, y_train, position, X_val, y_val, ...) -> bundle
    predict_fn: Callable        # (bundle, X, **kwargs) -> np.ndarray
    deps: list[str]             # meta-models only: base model names required
```

The `family` field controls which pass of the CV loop runs the model:

| Family | CV pass | Description |
|--------|---------|-------------|
| `tabular` | Pass 1 | Standard — fit on X_train, predict on X_val |
| `meta` | Pass 2 | Fit on base-model prediction columns, not raw features |
| `sequential` | Separate pipeline | Requires sequence reshaping; lives in evaluate_sequential.py |
| `decomposed` | Pass 1 | Treated as tabular externally; build_fn orchestrates sub-models |

The three existing files (`evaluate.py`, `train.py`, `predict.py`) are refactored to
iterate over the registry instead of hardcoded if/elif branches.

---

## Files

**Create:**
- `ml/models.py` — registry, ModelSpec dataclass, all build_fn and predict_fn definitions
- `ml/evaluate_sequential.py` — standalone CV pipeline for LSTM/GRU/TFT/N-BEATS (Batch 6)

**Modify:**
- `ml/evaluate.py` — replace hardcoded model blocks with registry iteration; add meta-model
  second pass; generalise summarise_cv, beats_baseline, and plot functions
- `ml/train.py` — replace VALID_MODELS tuple and elif dispatch with registry lookup
- `ml/predict.py` — replace VALID_MODELS tuple and elif dispatch with registry lookup

**Unchanged:**
- `ml/features.py` — feature contract is model-agnostic; no changes needed

---

## Dependencies

| Package | Required by | When to install |
|---------|-------------|-----------------|
| `xgboost` | XGBoost | Batch 2 |
| `statsmodels` | Poisson GLM | Batch 3 (likely already installed from EDA) |
| `catboost` | CatBoost | Batch 2 (optional, ~50 MB) |
| `torch` | LSTM, GRU, TFT | Batch 6 only |
| `pytorch-forecasting` | TFT, N-BEATS/N-HiTS | Batch 6 only — install only when needed |

All sklearn models (ElasticNet, Lasso, BayesianRidge, RandomForest, ExtraTrees,
HistGradientBoosting, MLP, LogisticRegression, PolynomialFeatures) are already available.

---

## Batch 0 — Architecture Refactor

No new models. Prerequisite for all subsequent batches.

### ml/models.py (create)

- Define `ModelSpec` dataclass
- Register the three existing models: `baseline`, `ridge`, `lgbm`
- Expose:
  - `get_registry() -> dict[str, ModelSpec]`
  - `get_model(name) -> ModelSpec`
  - `tabular_models() -> list[ModelSpec]`
  - `meta_models() -> list[ModelSpec]`
  - `sequential_models() -> list[ModelSpec]`

### ml/evaluate.py (refactor)

- Replace hardcoded baseline/ridge/lgbm blocks in `run_cv()` with:
  - Pass 1: loop over `tabular_models()` from registry
  - Pass 2: loop over `meta_models()` from registry (after Pass 1 completes per fold)
  - `oof_preds: dict[str, list]` accumulated across folds for stacking use in Pass 2
- `summarise_cv()` — replace hardcoded model list with registry names
- `beats_baseline()` — loop over all non-baseline registered models
- `plot_calibration()`, `plot_mae_by_fold()` — accept model name list, not hardcoded labels
- Remove `VALID_MODELS` tuple

### ml/train.py (refactor)

- Replace `VALID_MODELS` tuple with `get_registry()`
- `train_position()` dispatch: `spec = get_model(model_name); spec.build_fn(...)`
- Shared `_train_tabular()` wrapper replaces per-model train functions
- Remove per-model elif branches

### ml/predict.py (refactor)

- Remove `VALID_MODELS` tuple
- `predict_position()` dispatch: `spec = get_model(model_name); spec.predict_fn(...)`
- Remove per-model elif branches

**Verification gate:** Run `python -m ml.evaluate --position GK` before and after refactor.
CV MAE values must match to 4 decimal places.

---

## Batch 1 — Tier 1 Remaining + Linear Models

Models: `position_mean`, `elasticnet`, `bayesian_ridge`, `lasso`

All registered as `family='tabular'`. Linear models share the imputation + scaling pattern
with Ridge.

| Model | Tier | requires_imputation | requires_scaling | Notes |
|-------|:----:|:------------------:|:----------------:|-------|
| position_mean | 1 | False | False | Mean pts by (position, was_home); fit on training fold |
| elasticnet | 2 | True | True | l1_ratio=0.5, alpha=1.0; directly addresses xg/xa/xgi collinearity |
| bayesian_ridge | 2 | True | True | Stores pred_std for dashboard confidence bands |
| lasso | 3 | True | True | alpha=1.0; expect coefficient instability on correlated xG features |

Bundle additions:
- `bayesian_ridge`: add `pred_std: np.ndarray` (from `.predict(return_std=True)`)
  stored alongside `preds`; written to `preds_df` as `pred_std_bayesian_ridge`
- `position_mean`: `{'means': dict[(position, was_home), float], 'fallback': float}`

`_record_metrics` uses only the point estimate `preds` column — no changes needed there.

Dependencies: none new.

**Verification gate:** All 4 models pass the baseline gate (beat rolling-mean baseline on
≥2 of 3 primary metrics: MAE, RMSE, Spearman). BayesianRidge pred_std > 0 for all rows.

---

## Batch 2 — Gradient Boosting Expansion

Models: `xgb`, `random_forest`, `extra_trees`, `hist_gb`, `catboost` (optional)

| Model | Tier | requires_imputation | requires_scaling | NaN strategy |
|-------|:----:|:------------------:|:----------------:|--------------|
| xgb | 2 | False | False | Native via tree_method='hist' |
| random_forest | 2 | True | False | sklearn RF does NOT handle NaN natively |
| extra_trees | 3 | True | False | Same as RF |
| hist_gb | 3 | False | False | Native via missing_values=np.nan |
| catboost | 3 | False | False | Native |

XGBoost starting hyperparameters (position-specific, mirroring LightGBM conservative approach for GK):

| Position | max_depth | learning_rate | n_estimators | subsample |
|----------|:---------:|:-------------:|:------------:|:---------:|
| GK | 3 | 0.05 | 200 | 0.8 |
| DEF | 5 | 0.05 | 300 | 0.8 |
| MID | 5 | 0.05 | 300 | 0.8 |
| FWD | 5 | 0.05 | 300 | 0.8 |

All bundles: `{model, feature_cols, params, preds}` — same structure as the lgbm bundle.

Dependencies: `pip install xgboost` (catboost optional: `pip install catboost`).

**Verification gate:** XGBoost MAE within 5% of LightGBM MAE for all positions.
All models pass the baseline gate.

---

## Batch 3 — Probabilistic and Enhanced Baselines

Models: `poisson_glm`, `fdr_mean`, `last_season_avg`

### poisson_glm

- `statsmodels.api.GLM` with `family=sm.families.Poisson()`
- `total_points` can be negative (red card = −1); store `min_shift` in bundle computed
  from training fold only; add shift to target at fit time, subtract from predictions at inference
- `family='tabular', requires_imputation=True, requires_scaling=False`
- Bundle: `{model, feature_cols, season_means, global_means, min_shift, preds}`

### fdr_mean

- Non-ML model. Multiplies `pts_rolling_5gw` by a difficulty multiplier derived from
  `opponent_season_rank` bin (top 6 = bin A, ranks 7–14 = bin B, bottom 6 = bin C)
- Bin multipliers fit by minimising MAE on training fold per position
- Bundle: `{'multipliers': dict[str, float], 'fallback': float}`
- `family='tabular', requires_imputation=False, requires_scaling=False`

### last_season_avg

- For GW1 rows: look up player's prior-season pts_per_gw from training fold summary
- For all other GWs: fall back to `pts_rolling_5gw`
- Bundle: `{'player_prior_season': dict[(player_code, season_id), float], 'global_fallback': float}`
- Evaluation: compute GW1-specific MAE as a supplementary metric (cold-start focus)

Dependencies: `pip install statsmodels` (likely already present from EDA phase).

**Verification gate:** `fdr_mean` beats plain rolling-mean baseline on GK and DEF (most
fixture-sensitive positions). `last_season_avg` evaluated specifically on GW1 rows.

---

## Batch 4 — Neural and Meta-Ensemble Models

Models: `mlp`, `simple_avg`, `stacking`

### mlp

- `sklearn.neural_network.MLPRegressor`
- `family='tabular', requires_imputation=True, requires_scaling=True`
- Architecture: GK `(32, 16)`, others `(64, 32)` (tighter for GK due to row count)
- `activation='relu', max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42`

### simple_avg

- `family='meta', deps=['ridge', 'xgb', 'elasticnet', 'lgbm']`
- No fitting required; uniform weights initially
- Bundle: `{'base_models': [...], 'weights': np.ndarray}`
- `build_fn` receives `{model_name: preds_array}` dict for val-fold predictions (Pass 2)
- `predict_fn` takes same dict, returns weighted mean

### stacking

- `family='meta', deps=['ridge', 'xgb', 'elasticnet', 'lgbm', 'random_forest']`
- Meta-learner: Ridge (alpha=0.5)
- `oof_preds` dict accumulated in `run_cv()` across folds; meta-learner fit on
  concatenated OOF from folds 1+2, evaluated on fold 3 only
- Bundle: `{meta_model, scaler, base_models, preds}`
- Single-fold evaluation is acceptable for a Tier 3 model

Meta-model CV loop changes to `evaluate.py`:
- After Pass 1 completes for a fold, collect all tabular `preds` into `fold_preds` dict
- Pass 2 iterates over `meta_models()`, calling `spec.build_fn(oof_preds_so_far, fold_preds_val)`
- `oof_preds` accumulated fold-by-fold for stacking

Dependencies: sklearn only.

**Verification gate:** `simple_avg` MAE ≤ best single-model MAE for all positions.
Stacking result is informational only (single fold).

---

## Batch 5 — Decomposed Models + Tier 3 Standalone

### minutes_model (`decomposed_minutes`)

- `family='decomposed'` — treated as tabular externally; build_fn is two-stage internally
- Stage 1: `LogisticRegression` on rotation-signal features
  (`mins_rolling_3gw, season_starts_rate_to_date, value_lag1, transfers_in_lag1`) → P(starts)
- Stage 2a: `Ridge` on full feature set for rows where `starts=1` → E[pts | started]
- Stage 2b: `Ridge` on full feature set for rows where `starts=0` → E[pts | benched]
- Prediction: `P(starts) * E[pts | started] + (1 - P(starts)) * E[pts | benched]`
- Bundle:
  ```
  {clf, reg_started, reg_benched,
   scaler_clf, scaler_reg,
   season_means_clf, season_means_reg,
   global_means_clf, global_means_reg,
   feature_cols_clf, feature_cols_reg}
  ```

### component_model (`decomposed_components`)

- Sub-models: one Ridge per scoring component (goals, assists, cs, bonus)
- FPL scoring rules stored in bundle (position-specific):
  - GK/DEF: goal=6, assist=3, cs=6, bonus=1
  - MID: goal=5, assist=3, cs=1, bonus=1
  - FWD: goal=4, assist=3, cs=0, bonus=1
- Bundle: `{models: dict[str, Ridge], scalers, season_means, global_means, scoring_rules}`
- Prediction: sum of `component_pred * scoring_rule` across all components

### poly_ridge (Tier 3 standalone)

- `PolynomialFeatures(degree=2, interaction_only=True)` + `Ridge`
- `interaction_only=True` limits expansion to pairwise interactions only
- `family='tabular', requires_imputation=True, requires_scaling=True`
- Bundle includes fitted `PolynomialFeatures` transformer for inference

### blending (Tier 3 standalone)

- `family='meta'`
- Held-out blend set = fold 3 validation set
- Meta-learner fit on fold 1+2 val-fold base-model predictions, evaluated on fold 3
- Same structural limitation as stacking (single-fold evaluation)

Dependencies: sklearn only.

**Verification gate:** `minutes_model` improves on Ridge for MID and FWD (rotation-heavy
positions). `component_model` assessed for interpretability; raw MAE improvement optional.

---

## Batch 6 — Sequential Models (Tier 3, separate pipeline)

Models: `lstm`, `gru`, `tft`, `nbeats`

All implemented in `ml/evaluate_sequential.py`. Completely separate from the main CV loop.

### Sequence preparation

- Reshape feature matrix from `(n_observations, n_features)` to
  `(n_players, n_timesteps, n_features)` using player-season groupby
- Pad and mask sequences to uniform length; mask passed to model
- Reuses same `CV_FOLDS` definitions and `build_feature_matrix()` output unchanged

### LSTM / GRU

- PyTorch. Hidden size 64, 2 layers, dropout 0.2
- `family='sequential'`; registered in `models.py` but called only by `evaluate_sequential.py`

### TFT / N-BEATS

- `pytorch-forecasting` library
- Only implement if LSTM/GRU shows improvement over LightGBM on ≥2 positions

### Outputs

- `logs/training/cv_metrics_{pos}_seq.csv` (same schema as main CV metrics file)
- Enables direct comparison with tabular results

Dependencies: `pip install torch pytorch-forecasting` — install only when starting this batch.

**Verification gate:** LSTM/GRU beats LightGBM on ≥2 positions before proceeding to
TFT/N-BEATS.

---

## Key Implementation Rules

These apply to every model in every batch:

1. All fitting (model, scaler, imputer, hyperparameter tuning) uses training fold data only — never validation fold data
2. `stratified_impute()` is called inside each `build_fn` that requires it; the registry `requires_imputation` flag is documentation and dispatch guidance, not automatic preprocessing
3. Scaler fitted inside `build_fn`, stored in bundle, applied (not refitted) inside `predict_fn`
4. Every bundle must be pkl-serialisable — no lambda functions, no live DB connections
5. `feature_cols` always stored in bundle to ensure inference feature alignment
6. `random_state=42` everywhere for reproducibility
7. All CV metrics appended to `logs/training/cv_metrics_{pos}.csv` using the same schema — new model rows added, existing rows unchanged
8. `beats_baseline` gate applied to every non-baseline model in `summarise_cv`

---

## Rollout Sequence

```
Batch 0    Refactor evaluate.py + train.py + predict.py; create ml/models.py
           Gate: CV MAE values numerically identical before and after refactor

Batch 1    position_mean, elasticnet, bayesian_ridge, lasso
           Gate: all 4 pass baseline gate; bayesian_ridge pred_std > 0

Batch 2    xgb, random_forest, extra_trees, hist_gb, catboost (optional)
           Gate: xgb MAE within 5% of lgbm; all pass baseline gate

Batch 3    poisson_glm, fdr_mean, last_season_avg
           Gate: fdr_mean beats baseline on GK/DEF; last_season_avg on GW1 rows

Batch 4    mlp, simple_avg, stacking
           Gate: simple_avg MAE <= best single model; stacking informational

Batch 5    minutes_model, component_model, poly_ridge, blending
           Gate: minutes_model improves on Ridge for MID/FWD

Batch 6    evaluate_sequential.py: lstm/gru
           Gate: beats lgbm on >=2 positions before proceeding to tft/nbeats
```

Update the `Status` column in `revised_modelling_plan.md` after each batch completes.
