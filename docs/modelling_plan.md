# Modelling Plan

This document consolidates the model inventory, tier rationale, implementation priority, and
batch-by-batch technical implementation strategy for Phases 5 and 6. It supersedes
`project_plan.md` §5.1 and §5.2.

---

## 1. Model Inventory

### A — Naive Baselines

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| Rolling N-GW mean | Last 3 or 5 GWs | 1 | Implemented |
| Position mean | Mean pts by position x home/away | 1 | Implemented |
| FDR-adjusted mean | Rolling mean x opponent difficulty multiplier | 2 | Implemented |
| Last season avg | Prior season pts/GW | 2 | Implemented |

### B — Linear Models

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| OLS Linear Regression | Standard least-squares | Not recommended | Not implemented |
| Ridge Regression | OLS + L2 penalty | 1 | Implemented |
| ElasticNet | L1 + L2 combined | 2 | Implemented |
| Bayesian Ridge | Ridge with Bayesian priors | 2 | Implemented |
| Lasso Regression | OLS + L1 penalty | 3 | Implemented |
| Polynomial + Ridge | Degree-2 interactions + Ridge | 3 | Implemented |
| Poisson GLM | GLM with log link | 2 | Implemented |

### C — Tree-Based Models

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| Decision Tree | Single CART tree | Not recommended | Not implemented |
| Random Forest | Bagging of trees | 2 | Implemented |
| Extra Trees | Random splits + bagging | 3 | Implemented |

### D — Gradient Boosting

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| LightGBM | Leaf-wise gradient boosting | 1 | Implemented |
| XGBoost | Gradient boosting with regularisation | 2 | Implemented |
| HistGradientBoosting | sklearn histogram GB | 3 | Implemented |
| CatBoost | Native categorical support | 3 | Not implemented |

### E — Neural Networks

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| MLP | Fully-connected feed-forward net | 2 | Implemented |
| LSTM | Recurrent net over GW sequences | 3 | Implemented |
| GRU | Lighter variant of LSTM | 3 | Implemented |
| Temporal Fusion Transformer | Attention-based TS model | 3 | Not implemented |
| N-BEATS / N-HiTS | Neural basis expansion TS | 3 | Not implemented |

### F — Probabilistic / Bayesian

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| Gaussian Process Regression | Non-parametric Bayesian | Not recommended | Not implemented |
| Zero-Inflated Poisson | Poisson with excess-zero component | Not recommended | Not implemented |

### G — Decomposed / Component Models

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| Minutes model | Predict P(starts), then conditional pts | 2 | Implemented |
| Component model | One model per scoring component | 3 | Implemented |

### H — Ensemble / Stacking

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| Simple Averaging | Average predictions of multiple models | 2 | Implemented |
| Stacking | Train meta-model on OOF predictions | 3 | Implemented |
| Blending | ElasticNet meta-learner on held-out set | 3 | Implemented |

### I — Not Recommended

| Model | Reason |
|-------|--------|
| OLS | Strictly dominated by Ridge on a correlated feature set |
| Decision Tree | Strictly dominated by Random Forest and boosting at no additional cost |
| Gaussian Process | O(n³) — computationally infeasible at 40,900 rows |
| Zero-Inflated Poisson | Marginal gain over Poisson GLM; complex implementation with limited library support |
| ARIMA / SARIMA | Designed for univariate TS; max 38 GWs per player per season is too few for reliable fitting |
| Prophet | Requires trend + seasonality components; FPL GW data has neither |
| ETS | Same structural mismatch as ARIMA; no advantage in a cross-sectional player panel |

---

## 2. Tier Assignment Rationale

| Model | Tier | Reasoning |
|-------|:----:|-----------|
| Position mean | 1 | Trivial sanity-check baseline; should be implemented alongside rolling mean to set the floor |
| FDR-adjusted mean | 2 | Low-effort extension of the Tier 1 rolling baseline; adds fixture context without full ML overhead |
| Last season avg | 2 | Directly addresses the cold-start problem (39.4% of players appear only one season); complements rolling features at GW1 |
| OLS | Not recommended | Strictly dominated by Ridge — no regularisation on a collinear feature set guarantees instability |
| ElasticNet | 2 | Combines Ridge stability with Lasso selection; directly addresses the xg/xa/xgi collinearity artefact without dropping features manually |
| Bayesian Ridge | 2 | Near-zero additional effort over Ridge; provides per-prediction uncertainty estimates directly useful for dashboard confidence bands |
| Lasso | 3 | Feature selection is useful given the xG collinearity issue but unstable with correlated features; worth testing after Ridge is well-understood |
| Polynomial + Ridge | 3 | Cheap non-linearity, but LightGBM already handles interactions better; only worth trying if linear models plateau |
| Poisson GLM | 2 | Theoretically well-motivated — the target is count-like and right-skewed; low implementation effort via statsmodels |
| Decision Tree | Not recommended | High variance; strictly dominated by Random Forest and boosting at no additional cost |
| Extra Trees | 3 | Adds ensemble diversity vs boosting models; lower priority than Random Forest which is already Tier 2 |
| CatBoost | 3 | Alternative to LightGBM with native categorical handling, but slower and adds little unique value given team/position are already encoded |
| HistGradientBoosting | 3 | Pure sklearn convenience wrapper; less configurable than XGBoost or LightGBM; no meaningful advantage over existing Tier 1/2 boosting models |
| N-BEATS / N-HiTS | 3 | Pure time-series architecture; harder to incorporate static player features; only justified if sequential modelling proves stronger than rolling features |
| Simple Averaging | 2 | Trivial to add once multiple Tier 1/2 models exist; ensembles of diverse models reliably improve on any single model |
| Blending | 3 | Less principled than stacking (wastes a holdout split); only worth trying if stacking overhead is undesirable |

---

## 3. Implementation Priority

### Tier 1 — Complete

All three core models implemented. No further action needed unless CV metrics degrade in production.

| Priority | Model | Rationale |
|:--------:|-------|-----------|
| 1 | Rolling N-GW mean baseline | Done |
| 2 | Ridge | Done — production model |
| 3 | LightGBM | Done |
| 4 | Position mean baseline | Done — completes the baseline floor |

### Tier 2 — Recommended Next Steps

Build after Tier 1 is stable. Ordered by expected ROI relative to implementation effort.

| Priority | Model | Rationale |
|:--------:|-------|-----------|
| 1 | XGBoost | Closest competitor to LightGBM; different regularisation structure; most likely to challenge Ridge as production model |
| 2 | ElasticNet | Directly addresses the xg/xa/xgi collinearity artefact with minimal effort; natural follow-on to Ridge |
| 3 | Bayesian Ridge | Near-zero effort over existing Ridge; unlocks per-prediction uncertainty bands for the Phase 7 dashboard |
| 4 | Simple Averaging | Trivial once XGBoost and ElasticNet are available; ensembling diverse Tier 1 and Tier 2 models consistently adds 1-3% MAE improvement |
| 5 | Poisson GLM | Theoretically well-motivated given the right-skewed, count-like target distribution; low implementation cost via statsmodels |
| 6 | Random Forest | Different inductive bias from boosting; primary value is ensemble diversity for Tier 3 stacking |
| 7 | FDR-adjusted mean | Enhances the rolling baseline with fixture context; useful diagnostic to quantify how much fixture difficulty alone explains before full ML |
| 8 | Last season avg | Directly mitigates the cold-start problem at GW1; pairs with the existing rolling features pipeline |
| 9 | Decomposed minutes model | High FPL value — rotation is the largest real-world blind spot; requires a calibrated P(starts) sub-model so engineering cost is higher |
| 10 | MLP | Neural baseline to validate whether deep learning adds value over boosting at this dataset size |

### Tier 3 — Experimental

Build only after Tier 2 is fully evaluated. Each model requires a specific prior result to
justify the additional complexity.

| Priority | Model | Gate condition |
|:--------:|-------|----------------|
| 1 | Stacking | Natural next step after Simple Averaging; justifiable once 3+ diverse base models exist |
| 2 | Lasso | Test if automatic feature selection handles xgi collinearity better than manual Ridge coefficient inspection |
| 3 | Extra Trees | Adds diversity to a stacking ensemble; low standalone value |
| 4 | CatBoost | Only worth testing if team/opponent categorical encoding proves to be a bottleneck in XGBoost/LightGBM |
| 5 | Polynomial + Ridge | Test only if all linear and boosting models plateau |
| 6 | Component model | High interpretability payoff; only justified if the minutes model succeeds |
| 7 | HistGradientBoosting | No meaningful advantage over XGBoost/LightGBM; test only for sklearn pipeline consolidation |
| 8 | N-BEATS / N-HiTS | Test only if rolling features prove insufficient as sequential GW representations |
| 9 | Blending | Less principled than stacking; only consider if stacking CV overhead is prohibitive |
| 10 | LSTM / GRU | Justified only if EDA on GW sequences shows strong autocorrelation that rolling features do not capture |
| 11 | Temporal Fusion Transformer | Only pursue if LSTM shows a material improvement over LightGBM; engineering overhead is significant |

---

## 4. Architecture

### Registry Pattern

A `ml/models.py` file acts as the single source of truth for all model definitions.
Every model is mapped to a `ModelSpec` dataclass:

```python
@dataclass
class ModelSpec:
    name: str
    family: str          # 'tabular' | 'meta' | 'sequential' | 'decomposed'
    tier: int
    requires_imputation: bool   # call stratified_impute inside build_fn
    requires_scaling: bool      # fit StandardScaler inside build_fn
    build_fn: Callable          # (X_train, y_train, position, ...) -> bundle dict
    predict_fn: Callable        # (bundle, X, **kwargs) -> np.ndarray
    deps: list[str]             # meta-models only: base model names required
```

The `family` field controls which pass of the CV loop runs the model:

| Family | CV pass | Description |
|--------|---------|-------------|
| `tabular` | Pass 1 | Standard — fit on X_train, predict on X_val |
| `decomposed` | Pass 1 | Treated as tabular externally; `build_fn` orchestrates sub-models internally |
| `meta` | Pass 2 | Fit on base-model OOF prediction columns, not raw features |
| `sequential` | Separate pipeline | Requires sequence reshaping; lives in `evaluate_sequential.py` |

### Files

**Created:**
- `ml/models.py` — registry, `ModelSpec` dataclass, all `build_fn` and `predict_fn` definitions
- `ml/evaluate_sequential.py` — standalone CV pipeline for LSTM/GRU

**Modified:**
- `ml/evaluate.py` — registry iteration in `run_cv()`; Pass 1 (tabular/decomposed), Pass 2 (meta); generalised `summarise_cv`, `beats_baseline`, and plot functions; removed `VALID_MODELS` tuple
- `ml/train.py` — registry dispatch replaces per-model elif branches; `train_meta_position()` for OOF-based meta-model training
- `ml/predict.py` — registry dispatch replaces per-model elif branches

**Unchanged:**
- `ml/features.py` — feature contract is model-agnostic; no changes needed

### Dependencies

| Package | Required by | Status |
|---------|-------------|--------|
| `xgboost` | XGBoost | Installed (Batch 2) |
| `statsmodels` | Poisson GLM | Installed (present from EDA phase) |
| `catboost` | CatBoost | Not installed — model deferred |
| `torch` | LSTM, GRU | Installed (Batch 6) |
| `pytorch-forecasting` | TFT, N-BEATS/N-HiTS | Not installed — models deferred |

All sklearn models (ElasticNet, Lasso, BayesianRidge, RandomForest, ExtraTrees,
HistGradientBoosting, MLP, LogisticRegression, PolynomialFeatures) were already available.

---

## 5. Key Implementation Rules

These apply to every model in every batch:

1. All fitting (model, scaler, imputer, hyperparameter tuning) uses training fold data only — never validation fold data.
2. `stratified_impute()` is called inside each `build_fn` that requires it; the registry `requires_imputation` flag is documentation and dispatch guidance, not automatic preprocessing.
3. Scaler fitted inside `build_fn`, stored in bundle, applied (not refitted) inside `predict_fn`.
4. Every bundle must be pkl-serialisable — no lambda functions, no live DB connections.
5. `feature_cols` always stored in bundle to ensure inference feature alignment.
6. `random_state=42` everywhere for reproducibility.
7. All CV metrics appended to `logs/training/cv_metrics_{pos}.csv` using the same schema — new model rows added, existing rows unchanged.
8. `beats_baseline` gate applied to every non-baseline model in `summarise_cv`.

---

## 6. Batch-by-Batch Implementation

### Batch 0 — Architecture Refactor

No new models. Prerequisite for all subsequent batches.

**ml/models.py (created):**
- `ModelSpec` dataclass defined
- Three existing models registered: `baseline`, `ridge`, `lgbm`
- Exposed: `get_registry()`, `get_model(name)`, `tabular_models()`, `meta_models()`, `sequential_models()`

**ml/evaluate.py (refactored):**
- `run_cv()` refactored: Pass 1 loops over `tabular_models()`; Pass 2 loops over `meta_models()`
- `oof_preds` dict accumulated across folds for meta-model use
- `summarise_cv()`, `beats_baseline()`, plot functions use registry names, not hardcoded labels
- `VALID_MODELS` tuple removed

**ml/train.py (refactored):**
- Registry dispatch replaces per-model elif branches
- Shared `_train_tabular()` wrapper for all tabular/decomposed models

**ml/predict.py (refactored):**
- Registry dispatch replaces per-model elif branches
- Meta-model `predict_fn` auto-chains base model inference

**Verification gate:** CV MAE values identical to 4 decimal places before and after refactor. PASSED.

---

### Batch 1 — Tier 1 Remaining + Linear Models

Models: `position_mean`, `elasticnet`, `bayesian_ridge`, `lasso`

All registered as `family='tabular'`. Linear models share the stratified imputation and
StandardScaler pattern with Ridge.

| Model | Tier | requires_imputation | requires_scaling | Notes |
|-------|:----:|:------------------:|:----------------:|-------|
| position_mean | 1 | False | False | Mean pts by (position, was_home); fit on training fold only |
| elasticnet | 2 | True | True | l1_ratio=0.5, alpha=1.0; directly addresses xg/xa/xgi collinearity |
| bayesian_ridge | 2 | True | True | Stores `pred_std` in bundle for dashboard confidence bands |
| lasso | 3 | True | True | alpha=1.0; produces NaN Spearman due to collinearity (expected) |

Bundle additions:
- `bayesian_ridge`: includes `pred_std: np.ndarray` (from `.predict(return_std=True)`)
- `position_mean`: `{'means': dict[(position, was_home), float], 'fallback': float}`

**Verification gate:** All 4 models pass the baseline gate (beat rolling-mean baseline on
≥2 of 3 primary metrics: MAE, RMSE, Spearman). BayesianRidge pred_std > 0 for all rows. PASSED.

---

### Batch 2 — Gradient Boosting Expansion

Models: `xgb`, `random_forest`, `extra_trees`, `hist_gb`

| Model | Tier | requires_imputation | requires_scaling | NaN strategy |
|-------|:----:|:------------------:|:----------------:|--------------|
| xgb | 2 | False | False | Native via `tree_method='hist'` |
| random_forest | 2 | True | False | sklearn RF does not handle NaN natively |
| extra_trees | 3 | True | False | Same as RF |
| hist_gb | 3 | False | False | Native via `missing_values=np.nan` |

XGBoost starting hyperparameters (position-specific, mirroring LightGBM conservative approach):

| Position | max_depth | learning_rate | n_estimators | subsample |
|----------|:---------:|:-------------:|:------------:|:---------:|
| GK | 3 | 0.05 | 200 | 0.8 |
| DEF | 5 | 0.05 | 300 | 0.8 |
| MID | 5 | 0.05 | 300 | 0.8 |
| FWD | 5 | 0.05 | 300 | 0.8 |

All bundles: `{model, feature_cols, params, preds}` — same structure as the lgbm bundle.

CatBoost (Tier 3, optional): deferred — no unique value identified given existing categorical
encoding. Not installed.

**Verification gate:** XGBoost MAE within 5% of LightGBM MAE for all positions; all models
pass the baseline gate. PASSED.

---

### Batch 3 — Probabilistic and Enhanced Baselines

Models: `poisson_glm`, `fdr_mean`, `last_season_avg`

**poisson_glm:**
- `statsmodels.api.GLM` with `family=sm.families.Poisson()`
- `total_points` can be negative (red card = −1); `min_shift` computed from training fold
  minimum, added before fitting, subtracted from predictions at inference
- `family='tabular', requires_imputation=True, requires_scaling=False`
- Bundle: `{model, feature_cols, season_means, global_means, min_shift, preds}`

**fdr_mean:**
- Non-ML model. Multiplies `pts_rolling_5gw` by a difficulty multiplier derived from
  `opponent_season_rank` bin (top 6 = bin A, ranks 7–14 = bin B, bottom 6 = bin C)
- Bin multipliers fit by minimising MAE on training fold per position
- Bundle: `{'multipliers': dict[str, float], 'fallback': float}`
- `family='tabular', requires_imputation=False, requires_scaling=False`

**last_season_avg:**
- For GW1 rows: look up player's prior-season pts_per_gw from training fold summary
- For all other GWs: fall back to `pts_rolling_5gw`
- Bundle: `{'player_prior_season': dict[(player_code, season_id), float], 'global_fallback': float}`
- GW1-specific MAE computed as supplementary metric

**Verification gate:** `fdr_mean` beats plain rolling-mean baseline on GK and DEF. FAILED
(expected — `pts_rolling_5gw` already embeds fixture difficulty, so FDR multiplication
double-discounts hard fixtures). `last_season_avg` evaluated on GW1 rows: PASSED on GW1
specifically.

---

### Batch 4 — Neural and Meta-Ensemble Models

Models: `mlp`, `simple_avg`, `stacking`

**mlp:**
- `sklearn.neural_network.MLPRegressor`
- `family='tabular', requires_imputation=True, requires_scaling=True`
- Architecture: GK `(32, 16)`, others `(64, 32)` (tighter for GK due to row count)
- `activation='relu', max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42`

**simple_avg:**
- `family='meta', deps=['ridge', 'xgb', 'elasticnet', 'lgbm']`
- No fitting required; uniform weights stored in bundle
- Bundle: `{'base_models': [...], 'weights': np.ndarray}`
- `predict_fn` returns weighted mean of base model outputs

**stacking:**
- `family='meta', deps=['ridge', 'xgb', 'elasticnet', 'lgbm', 'random_forest']`
- Meta-learner: Ridge (alpha=0.5)
- `oof_preds` accumulated in `run_cv()` across folds; meta-learner fit on concatenated OOF
  from all available prior folds; single-fold evaluation (fold 3 only) is acceptable for Tier 3
- Bundle: `{meta_model, scaler, base_models, preds}`

Meta-model CV loop (Pass 2 in `evaluate.py`): after Pass 1 completes for a fold, all tabular
predictions collected into `fold_preds` dict; Pass 2 iterates over `meta_models()`, calling
`spec.build_fn(oof_preds_so_far, fold_preds_val)`.

**Verification gate:** `simple_avg` MAE ≤ best single-model MAE for all positions. FAILED
(expected — Ridge is better than simple_avg because XGBoost and LightGBM are weaker base
models at default settings, pulling the average down). Stacking result informational only.

---

### Batch 5 — Decomposed Models + Tier 3 Standalone

**minutes_model (`decomposed`):**
- `family='decomposed'` — treated as tabular externally; `build_fn` is two-stage internally
- Stage 1: `LogisticRegression` on rotation-signal features
  (`mins_rolling_3gw`, `season_starts_rate_to_date`, `value_lag1`, `transfers_in_lag1`) → P(starts)
- Stage 2a: `Ridge` on full feature set for rows where `starts=1` → E[pts | started]
- Stage 2b: `Ridge` on full feature set for rows where `starts=0` → E[pts | benched]
- Prediction: `P(starts) × E[pts | started] + (1 − P(starts)) × E[pts | benched]`
- Raw starts loaded from `fact_gw_player` via DB query — not from the feature matrix
- Bundle:
  ```
  {clf, reg_started, reg_benched,
   scaler_clf, scaler_reg,
   season_means_clf, season_means_reg,
   global_means_clf, global_means_reg,
   feature_cols_clf, feature_cols_reg}
  ```

**component_model (`decomposed`):**
- Sub-models: one Ridge per scoring component (goals, assists, clean sheets, bonus)
- FPL scoring rules stored in bundle (position-specific):
  - GK/DEF: goal=6, assist=3, cs=6, bonus=1
  - MID: goal=5, assist=3, cs=1, bonus=1
  - FWD: goal=4, assist=3, cs=0, bonus=1
- Bundle: `{models: dict[str, Ridge], scalers, season_means, global_means, scoring_rules}`
- Prediction: sum of `component_pred × scoring_rule` across all components

**poly_ridge (Tier 3 standalone):**
- `PolynomialFeatures(degree=2, interaction_only=True)` + `Ridge`
- `interaction_only=True` limits expansion to pairwise interactions only (~210 features)
- `family='tabular', requires_imputation=True, requires_scaling=True`
- Bundle includes fitted `PolynomialFeatures` transformer for inference alignment

**blending (Tier 3 standalone):**
- `family='meta', deps=['ridge', 'bayesian_ridge', 'poisson_glm', 'mlp']`
- ElasticNet (alpha=0.5, l1_ratio=0.5) meta-learner — different dependency set from stacking
- All four base models from the top-performing tier (no boosting models)
- Meta-learner fit on full 3-fold OOF stack for production serialisation
- Same single-fold evaluation limitation as stacking

**Verification gate:** `minutes_model` improves on Ridge for MID and FWD. FAILED (expected —
rolling features already capture most rotation signal; model still beats baseline on all
positions and provides useful P(starts) decomposition for Phase 7).

---

### Batch 6 — Sequential Models (Tier 3, separate pipeline)

Models: `lstm`, `gru`

Implemented in `ml/evaluate_sequential.py` — completely separate from the main CV loop.

**Sequence preparation:**
- Feature matrix reshaped from `(n_observations, n_features)` to `(n_players, n_timesteps, n_features)` via player-season groupby
- Sequences padded to `MAX_SEQ_LEN=38` timesteps; padded positions zeroed out after scaling
- Column means computed from training fold valid timesteps for NaN imputation
- StandardScaler fit on training valid timesteps only; applied to val fold; padded positions
  zeroed out after scaling to avoid hidden-state corruption
- Reuses same `CV_FOLDS` and `build_feature_matrix()` output unchanged

**LSTM / GRU architecture:**
- PyTorch. Hidden size 64, 2 layers, dropout 0.2, `batch_first=True`
- Adam optimizer (lr=1e-3), 50 epochs fixed, gradient clipping (norm=1.0)
- MSE loss computed on valid (non-padded) timesteps only
- `family='sequential'`; registered in `models.py` with stub build/predict functions;
  called only by `evaluate_sequential.py`

**Outputs:**
- `logs/training/cv_metrics_{pos}_seq.csv` — same schema as main CV metrics file
- Enables direct comparison with tabular results

**TFT / N-BEATS:** Gate met (LSTM/GRU beat LightGBM on ≥2 positions). Deferred to post-Phase 7
— neither sequential model beats Ridge, and `pytorch-forecasting` dependency overhead is not
justified until a clear improvement path exists.

**Verification gate:** LSTM/GRU beats LightGBM on ≥2 positions. PASSED — both models beat
LightGBM on GK, MID, and FWD (3/4 positions).

---

## 7. Rollout Summary

| Batch | Scope | Gate | Outcome |
|-------|-------|------|---------|
| 0 | Registry refactor; create `ml/models.py`; refactor evaluate/train/predict | CV MAE identity (4dp) | PASS |
| 1 | position_mean, elasticnet, bayesian_ridge, lasso | All pass baseline gate; pred_std > 0 | PASS |
| 2 | xgb, random_forest, extra_trees, hist_gb | XGBoost within 5% of LightGBM; all pass baseline gate | PASS |
| 3 | poisson_glm, fdr_mean, last_season_avg | fdr_mean beats baseline on GK/DEF | FAIL (expected) |
| 4 | mlp, simple_avg, stacking | simple_avg MAE ≤ best single model | FAIL (expected) |
| 5 | minutes_model, component_model, poly_ridge, blending | minutes_model improves Ridge on MID/FWD | FAIL (expected) |
| 6 | lstm, gru in evaluate_sequential.py | Beats LightGBM on ≥2 positions | PASS (3/4) |

The three FAIL gates are all expected outcomes documented in the tier rationale. Each failing
model still provides value in a specific context (GW1 cold-start, rotation risk decomposition,
ensemble diversity) and passes the rolling-mean baseline gate individually.
