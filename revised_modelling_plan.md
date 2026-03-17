# Revised Modelling Plan

This document supersedes the model brainstorm in `project_plan.md` §5.1 and §5.2.
All models from the original brainstorm have been assigned a tier. Tier 1 is unchanged
and reflects what has already been implemented. Tiers 2 and 3 are ordered by priority.

For the technical implementation strategy (registry architecture, batch sequencing,
per-model bundle specs, and verification gates) see `revised_modelling_implementation_plan.md`.

---

## Model Inventory

### A — Naive Baselines

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| Position mean | Mean pts by position x home/away | 1 | Implemented |
| Rolling N-GW mean | Last 3 or 5 GWs | 1 | Implemented |
| FDR-adjusted mean | Rolling mean x opponent difficulty multiplier | 2 | Implemented |
| Last season avg | Prior season pts/GW | 2 | Implemented |

### B — Linear Models

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| OLS Linear Regression | Standard least-squares | Not recommended | Not implemented |
| Ridge Regression | OLS + L2 penalty | 1 | Implemented |
| Lasso Regression | OLS + L1 penalty | 3 | Implemented |
| ElasticNet | L1 + L2 combined | 2 | Implemented |
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
| XGBoost | Gradient boosting with regularisation | 2 | Implemented |
| LightGBM | Leaf-wise gradient boosting | 1 | Implemented |
| CatBoost | Native categorical support | 3 | Not implemented |
| HistGradientBoosting | sklearn histogram GB | 3 | Implemented |

### E — Neural Networks

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| MLP | Fully-connected feed-forward net | 2 | Implemented |
| LSTM | Recurrent net over GW sequences | 3 | Not implemented |
| GRU | Lighter variant of LSTM | 3 | Not implemented |
| Temporal Fusion Transformer | Attention-based TS model | 3 | Not implemented |
| N-BEATS / N-HiTS | Neural basis expansion TS | 3 | Not implemented |

### F — Probabilistic / Bayesian

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| Bayesian Ridge | Ridge with Bayesian priors | 2 | Implemented |
| Gaussian Process Regression | Non-parametric Bayesian | Not recommended | Not implemented |
| Zero-Inflated Poisson | Poisson with excess-zero component | Not recommended | Not implemented |

### G — Decomposed / Component Models

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| Goals + Assists + CS + Bonus separately | One model per scoring component | 3 | Implemented |
| Minutes model first | Predict P(starts), then conditional pts | 2 | Implemented |

### H — Ensemble / Stacking

| Model | Description | Tier | Status |
|-------|-------------|:----:|--------|
| Simple Averaging | Average predictions of multiple models | 2 | Implemented |
| Stacking (meta-learner) | Train meta-model on OOF predictions | 3 | Implemented |
| Blending | Average on a held-out set | 3 | Implemented |

### I — Not Recommended

| Model | Reason | Tier | Status |
|-------|--------|:----:|--------|
| ARIMA / SARIMA | Too few per-player GWs; cross-sectional panel, not univariate TS | Not recommended | Not implemented |
| Prophet | Designed for trend + seasonality; FPL has neither | Not recommended | Not implemented |
| Exponential Smoothing (ETS) | Same data-length problem as ARIMA | Not recommended | Not implemented |

---

## Tier Assignment Rationale

| Model | Tier | Reasoning |
|-------|:----:|-----------|
| Position mean | 1 | Trivial sanity-check baseline; should be implemented alongside rolling mean to set the floor |
| FDR-adjusted mean | 2 | Low-effort extension of the Tier 1 rolling baseline; adds fixture context without full ML overhead |
| Last season avg | 2 | Directly addresses the cold-start problem (39.4% of players appear only one season); complements rolling features at GW1 |
| OLS | Not recommended | Strictly dominated by Ridge — no regularisation on a collinear feature set guarantees instability |
| Lasso | 3 | Feature selection is useful given the xG collinearity issue flagged in the modelling report, but unstable with correlated features; worth testing after Ridge is well-understood |
| ElasticNet | 2 | Combines Ridge stability with Lasso selection; directly addresses the xg/xa/xgi collinearity artefact without dropping features manually |
| Polynomial + Ridge | 3 | Cheap non-linearity, but LightGBM already handles interactions better; only worth trying if linear models plateau |
| Poisson GLM | 2 | Theoretically well-motivated — the target is count-like and right-skewed; low implementation effort via statsmodels |
| Decision Tree | Not recommended | High variance; strictly dominated by Random Forest and boosting at no additional cost |
| Extra Trees | 3 | Adds ensemble diversity vs boosting models; lower priority than Random Forest which is already Tier 2 |
| CatBoost | 3 | Alternative to LightGBM with native categorical handling, but slower and adds little unique value given team/position are already encoded |
| HistGradientBoosting | 3 | Pure sklearn convenience wrapper; less configurable than XGBoost or LightGBM; no meaningful advantage over existing Tier 1/2 boosting models |
| N-BEATS / N-HiTS | 3 | Pure time-series architecture; harder to incorporate static player features (price, position); only justified if sequential GW modelling proves stronger than rolling features |
| Bayesian Ridge | 2 | Near-zero additional effort over Ridge; provides per-prediction uncertainty estimates that are directly useful for dashboard confidence bands |
| Simple Averaging | 2 | Trivial to add once multiple Tier 1/2 models exist; ensembles of diverse models reliably improve on any single model |
| Blending | 3 | Less principled than stacking (wastes a holdout split); only worth trying if stacking overhead is undesirable |

---

## Implementation Priority List

### Tier 1 — Complete

All three models implemented. No further action needed unless CV metrics degrade in production.

| Priority | Model | Rationale |
|:--------:|-------|-----------|
| 1 | Rolling N-GW mean baseline | Done |
| 2 | Ridge | Done — production model |
| 3 | LightGBM | Done |
| 4 | Position mean baseline | Trivial to add; completes the baseline floor |

### Tier 2 — Recommended Next Steps

Build these after Tier 1 is stable. Ordered by expected ROI relative to implementation effort.

| Priority | Model | Rationale |
|:--------:|-------|-----------|
| 1 | XGBoost | Closest competitor to LightGBM; different regularisation structure; most likely to challenge Ridge as production model |
| 2 | ElasticNet | Directly addresses the xg/xa/xgi collinearity artefact flagged in the modelling report with minimal effort; natural follow-on to Ridge |
| 3 | Bayesian Ridge | Near-zero effort over existing Ridge; unlocks per-prediction uncertainty bands for the Phase 7 dashboard |
| 4 | Simple Averaging | Trivial once XGBoost and ElasticNet are available; ensembling diverse Tier 1 and Tier 2 models consistently adds 1-3% MAE improvement |
| 5 | Poisson GLM | Theoretically well-motivated given the right-skewed, count-like target distribution; low implementation cost via statsmodels |
| 6 | Random Forest | Different inductive bias from boosting; primary value is ensemble diversity for Tier 3 stacking |
| 7 | FDR-adjusted mean | Enhances the rolling baseline with fixture context; useful diagnostic to quantify how much fixture difficulty alone explains before full ML |
| 8 | Last season avg | Directly mitigates the cold-start problem at GW1; pairs with the existing rolling features pipeline |
| 9 | Decomposed minutes model | High FPL value — rotation is the largest real-world blind spot; requires a calibrated P(starts) sub-model so engineering cost is higher than the above |
| 10 | MLP | Neural baseline to validate whether deep learning adds value over boosting at this dataset size; low expectation but important to confirm empirically |

### Tier 3 — Experimental

Build only after Tier 2 is fully evaluated. Each model here requires a specific prior result
to justify the additional complexity.

| Priority | Model | Gate condition |
|:--------:|-------|----------------|
| 1 | Stacking (meta-learner) | Natural next step after Simple Averaging; justifiable once 3+ diverse base models exist |
| 2 | Lasso | Test if automatic feature selection handles xgi collinearity better than manual Ridge coefficient inspection |
| 3 | Extra Trees | Adds diversity to a stacking ensemble; low standalone value |
| 4 | CatBoost | Only worth testing if team/opponent categorical encoding proves to be a bottleneck in XGBoost/LightGBM |
| 5 | Polynomial + Ridge | Test only if all linear and boosting models plateau; LightGBM likely already captures interaction effects |
| 6 | Decomposed component models | High interpretability payoff; only justified if the minutes model (Tier 2 priority 9) succeeds |
| 7 | HistGradientBoosting | No meaningful advantage over XGBoost/LightGBM; test only for sklearn pipeline consolidation |
| 8 | N-BEATS / N-HiTS | Test only if rolling features prove insufficient as sequential GW representations |
| 9 | Blending | Less principled than stacking; only consider if stacking CV overhead is prohibitive |
| 10 | LSTM / GRU | Justified only if EDA on GW sequences shows strong autocorrelation that rolling features do not capture |
| 11 | Temporal Fusion Transformer | Only pursue if LSTM shows a material improvement over LightGBM; engineering overhead is significant |

### Not Recommended — Exclude

| Model | Reason |
|-------|--------|
| OLS | Strictly dominated by Ridge on a correlated feature set |
| Decision Tree | Strictly dominated by Random Forest and boosting at no additional cost |
| Gaussian Process | O(n^3) — computationally infeasible at 40,900 rows |
| Zero-Inflated Poisson | Marginal gain over Poisson GLM; complex implementation with limited library support |
| ARIMA / SARIMA | Designed for univariate TS; max 38 GWs per player per season is too few for reliable fitting |
| Prophet | Requires trend + seasonality components; FPL GW data has neither |
| ETS | Same structural mismatch as ARIMA; no advantage in a cross-sectional player panel |
