"""
Model registry for the FPL GW-level prediction pipeline.

Every model is represented by a ModelSpec dataclass that carries its build and
predict callables alongside metadata. The registry is the single source of truth
for all model definitions, replacing the hardcoded if/elif branches that existed
in evaluate.py, train.py, and predict.py.

Build function signature (all registered models must conform):
    build_fn(X_train, y_train, position, X_val=None, y_val=None,
             sid_train=None, sid_val=None, **kwargs) -> bundle dict

Predict function signature:
    predict_fn(bundle, X, sid=None, **kwargs) -> np.ndarray

Bundle contract (all tabular models):
    Required keys : feature_cols, preds (None when no val set provided)
    Optional keys : model, scaler, season_means, global_means, pred_std

Usage:
    from ml.models import get_registry, get_model, tabular_models, meta_models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    family: str              # 'tabular' | 'meta' | 'sequential' | 'decomposed'
    tier: int
    requires_imputation: bool
    requires_scaling: bool
    build_fn: Callable       # (X_train, y_train, position, X_val, y_val, sid_train, sid_val, **kw) -> bundle
    predict_fn: Callable     # (bundle, X, sid=None, **kw) -> np.ndarray
    deps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Build and predict implementations for Tier 1 models
#
# Deferred imports from ml.evaluate inside each function to avoid circular
# import: evaluate.py imports from models.py at module load; models.py must
# not import from evaluate.py at module load.
# ---------------------------------------------------------------------------

def _build_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from ml.evaluate import baseline_predict
    fallback_mean = float(y_train.mean())
    preds = baseline_predict(X_val, fallback_mean) if X_val is not None else None
    return {
        'fallback_mean': fallback_mean,
        'feature_cols':  list(X_train.columns),
        'preds':         preds,
    }


def _predict_baseline(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    fallback = bundle.get('fallback_mean', 1.0)
    return X['pts_rolling_5gw'].fillna(fallback).values


def _build_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from ml.evaluate import build_ridge
    return build_ridge(X_train, y_train, sid_train, X_val, sid_val)


def _predict_scaled_linear(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Shared inference for all imputed + scaled linear models
    (ridge, elasticnet, bayesian_ridge, lasso).
    Applies stored season/global means for imputation, then transforms with
    the stored scaler before calling model.predict().
    """
    feat_cols    = bundle['feature_cols']
    season_means = bundle['season_means']
    global_means = bundle['global_means']
    scaler       = bundle['scaler']
    model        = bundle['model']

    X_aligned = X[feat_cols].reset_index(drop=True)
    X_arr     = X_aligned.values.astype(float)
    gm        = global_means[feat_cols].values

    if sid is not None:
        s_vals = sid.reset_index(drop=True).values
        fill   = np.empty_like(X_arr)
        for i, s in enumerate(s_vals):
            if s in season_means.index:
                fill[i] = season_means.loc[s, feat_cols].values
            else:
                fill[i] = gm
        fill = np.where(np.isnan(fill), gm, fill)
    else:
        fill = np.tile(gm, (len(X_arr), 1))

    nan_mask = np.isnan(X_arr)
    X_filled = np.where(nan_mask, fill, X_arr)
    X_filled = np.where(np.isnan(X_filled), gm, X_filled)
    return model.predict(scaler.transform(X_filled))


def _predict_ridge(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    return _predict_scaled_linear(bundle, X, sid=sid)


def _build_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    tune: bool = False,
    extra_params: dict | None = None,
    **kwargs,
) -> dict:
    from ml.evaluate import build_lgbm
    return build_lgbm(
        X_train, y_train, position, X_val, y_val,
        tune=tune, extra_params=extra_params,
    )


def _predict_lgbm(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    return bundle['model'].predict(X[bundle['feature_cols']])


# ---------------------------------------------------------------------------
# Shared helper: impute + scale + fit linear estimator (Batch 1+)
# ---------------------------------------------------------------------------

def _build_scaled_linear(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    return_std: bool = False,
) -> dict:
    """
    Generic builder for any sklearn linear estimator that requires
    stratified mean imputation and StandardScaler.

    Parameters
    ----------
    estimator  : fitted in-place; must expose .fit(X, y) and .predict(X)
    return_std : if True, call .predict(X, return_std=True) on the val set
                 and store the uncertainty array as 'pred_std' in the bundle
                 (BayesianRidge only)
    """
    from ml.evaluate import stratified_impute
    from sklearn.preprocessing import StandardScaler

    if X_val is not None and sid_val is not None:
        X_tr_f, X_v_f, season_means, global_means = stratified_impute(
            X_train, X_val, sid_train, sid_val
        )
    else:
        X_tr_f, _, season_means, global_means = stratified_impute(
            X_train, X_train, sid_train, sid_train
        )
        X_v_f = None

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr_f)
    estimator.fit(X_tr_s, y_train.values)

    preds    = None
    pred_std = None
    if X_v_f is not None:
        X_v_s = scaler.transform(X_v_f)
        if return_std:
            preds, pred_std = estimator.predict(X_v_s, return_std=True)
        else:
            preds = estimator.predict(X_v_s)

    bundle = {
        'model':        estimator,
        'scaler':       scaler,
        'season_means': season_means,
        'global_means': global_means,
        'feature_cols': list(X_train.columns),
        'preds':        preds,
    }
    if pred_std is not None:
        bundle['pred_std'] = pred_std
    return bundle


# ---------------------------------------------------------------------------
# Batch 1: position_mean, elasticnet, bayesian_ridge, lasso
# ---------------------------------------------------------------------------

def _build_position_mean(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    """
    Fit home/away mean points for the given position on the training fold.
    Bundle: {'means': {(position, 0|1): float}, 'fallback': float, 'position': str}
    """
    was_home_tr = X_train['was_home'].fillna(0).astype(int)
    means: dict[tuple[str, int], float] = {}
    for home_val in (0, 1):
        mask = was_home_tr == home_val
        if mask.any():
            means[(position, home_val)] = float(y_train[mask].mean())
    fallback = float(y_train.mean())

    preds = None
    if X_val is not None:
        was_home_v = X_val['was_home'].fillna(0).astype(int).values
        preds = np.array([means.get((position, int(h)), fallback) for h in was_home_v])

    return {
        'means':        means,
        'fallback':     fallback,
        'position':     position,
        'feature_cols': list(X_train.columns),
        'preds':        preds,
    }


def _predict_position_mean(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    means    = bundle['means']
    fallback = bundle['fallback']
    position = bundle.get('position', '')
    was_home = X['was_home'].fillna(0).astype(int).values
    return np.array([means.get((position, int(h)), fallback) for h in was_home])


def _build_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from sklearn.linear_model import ElasticNet
    estimator = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
    return _build_scaled_linear(
        estimator, X_train, y_train, position, X_val, y_val, sid_train, sid_val
    )


def _build_bayesian_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from sklearn.linear_model import BayesianRidge
    estimator = BayesianRidge()
    return _build_scaled_linear(
        estimator, X_train, y_train, position, X_val, y_val, sid_train, sid_val,
        return_std=True,
    )


def _build_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from sklearn.linear_model import Lasso
    estimator = Lasso(alpha=1.0, random_state=42, max_iter=2000)
    return _build_scaled_linear(
        estimator, X_train, y_train, position, X_val, y_val, sid_train, sid_val
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    _REGISTRY[spec.name] = spec


_register(ModelSpec(
    name='baseline',
    family='tabular',
    tier=1,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_baseline,
    predict_fn=_predict_baseline,
))

_register(ModelSpec(
    name='ridge',
    family='tabular',
    tier=1,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_build_ridge,
    predict_fn=_predict_ridge,
))

_register(ModelSpec(
    name='lgbm',
    family='tabular',
    tier=1,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_lgbm,
    predict_fn=_predict_lgbm,
))

# Batch 1 ----------------------------------------------------------------

_register(ModelSpec(
    name='position_mean',
    family='tabular',
    tier=1,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_position_mean,
    predict_fn=_predict_position_mean,
))

_register(ModelSpec(
    name='elasticnet',
    family='tabular',
    tier=2,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_build_elasticnet,
    predict_fn=_predict_scaled_linear,
))

_register(ModelSpec(
    name='bayesian_ridge',
    family='tabular',
    tier=2,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_build_bayesian_ridge,
    predict_fn=_predict_scaled_linear,
))

_register(ModelSpec(
    name='lasso',
    family='tabular',
    tier=3,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_build_lasso,
    predict_fn=_predict_scaled_linear,
))

# ---------------------------------------------------------------------------
# Shared helper: impute only (no scaling) for sklearn tree ensembles (Batch 2)
# ---------------------------------------------------------------------------

def _build_imputed_tree(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
) -> dict:
    """
    Generic builder for tree models that need stratified mean imputation
    but no feature scaling (RandomForest, ExtraTrees).
    Fits and predicts with numpy arrays so sklearn does not store feature
    names and predict() does not raise a feature-name mismatch warning.
    """
    from ml.evaluate import stratified_impute

    if X_val is not None and sid_val is not None:
        X_tr_f, X_v_f, season_means, global_means = stratified_impute(
            X_train, X_val, sid_train, sid_val
        )
    else:
        X_tr_f, _, season_means, global_means = stratified_impute(
            X_train, X_train, sid_train, sid_train
        )
        X_v_f = None

    estimator.fit(X_tr_f.values, y_train.values)

    preds = None
    if X_v_f is not None:
        preds = estimator.predict(X_v_f.values)

    return {
        'model':        estimator,
        'season_means': season_means,
        'global_means': global_means,
        'feature_cols': list(X_train.columns),
        'preds':        preds,
    }


def _predict_imputed_tree(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Shared inference for imputed tree models (RF, ExtraTrees)."""
    feat_cols    = bundle['feature_cols']
    season_means = bundle['season_means']
    global_means = bundle['global_means']
    model        = bundle['model']

    X_aligned = X[feat_cols].reset_index(drop=True)
    X_arr     = X_aligned.values.astype(float)
    gm        = global_means[feat_cols].values

    if sid is not None:
        s_vals = sid.reset_index(drop=True).values
        fill   = np.empty_like(X_arr)
        for i, s in enumerate(s_vals):
            if s in season_means.index:
                fill[i] = season_means.loc[s, feat_cols].values
            else:
                fill[i] = gm
        fill = np.where(np.isnan(fill), gm, fill)
    else:
        fill = np.tile(gm, (len(X_arr), 1))

    nan_mask = np.isnan(X_arr)
    X_filled = np.where(nan_mask, fill, X_arr)
    X_filled = np.where(np.isnan(X_filled), gm, X_filled)
    return model.predict(X_filled)


def _predict_no_preproc(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Shared inference for native-NaN models (XGBoost, HistGB, CatBoost)."""
    return bundle['model'].predict(X[bundle['feature_cols']])


# ---------------------------------------------------------------------------
# Batch 2: gradient boosting expansion + sklearn tree ensembles
# ---------------------------------------------------------------------------

_XGB_BASE_PARAMS: dict[str, dict] = {
    'GK':  dict(max_depth=3, learning_rate=0.05, n_estimators=200, subsample=0.8),
    'DEF': dict(max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.8),
    'MID': dict(max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.8),
    'FWD': dict(max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.8),
}

_XGB_COMMON = dict(
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


def _build_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from xgboost import XGBRegressor
    params = {**_XGB_BASE_PARAMS[position], **_XGB_COMMON}
    model  = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val) if X_val is not None else None
    return {
        'model':        model,
        'feature_cols': list(X_train.columns),
        'params':       params,
        'preds':        preds,
    }


def _build_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    return _build_imputed_tree(
        estimator, X_train, y_train, position, X_val, y_val, sid_train, sid_val
    )


def _build_extra_trees(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from sklearn.ensemble import ExtraTreesRegressor
    estimator = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    return _build_imputed_tree(
        estimator, X_train, y_train, position, X_val, y_val, sid_train, sid_val
    )


def _build_hist_gb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val) if X_val is not None else None
    return {
        'model':        model,
        'feature_cols': list(X_train.columns),
        'preds':        preds,
    }


def _build_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6,
        random_seed=42, verbose=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val) if X_val is not None else None
    return {
        'model':        model,
        'feature_cols': list(X_train.columns),
        'preds':        preds,
    }


# Batch 2 registrations --------------------------------------------------

_register(ModelSpec(
    name='xgb',
    family='tabular',
    tier=2,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_xgb,
    predict_fn=_predict_no_preproc,
))

_register(ModelSpec(
    name='random_forest',
    family='tabular',
    tier=2,
    requires_imputation=True,
    requires_scaling=False,
    build_fn=_build_random_forest,
    predict_fn=_predict_imputed_tree,
))

_register(ModelSpec(
    name='extra_trees',
    family='tabular',
    tier=3,
    requires_imputation=True,
    requires_scaling=False,
    build_fn=_build_extra_trees,
    predict_fn=_predict_imputed_tree,
))

_register(ModelSpec(
    name='hist_gb',
    family='tabular',
    tier=3,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_hist_gb,
    predict_fn=_predict_no_preproc,
))

try:
    import catboost as _catboost_mod  # noqa: F401
    _register(ModelSpec(
        name='catboost',
        family='tabular',
        tier=3,
        requires_imputation=False,
        requires_scaling=False,
        build_fn=_build_catboost,
        predict_fn=_predict_no_preproc,
    ))
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_registry() -> dict[str, ModelSpec]:
    """Return a copy of the full model registry."""
    return dict(_REGISTRY)


def get_model(name: str) -> ModelSpec:
    """Return the ModelSpec for the given model name. Raises KeyError if absent."""
    if name not in _REGISTRY:
        raise KeyError(
            f'Model "{name}" not in registry. Available: {sorted(_REGISTRY)}'
        )
    return _REGISTRY[name]


def tabular_models() -> list[ModelSpec]:
    """All models with family='tabular', in registration order."""
    return [s for s in _REGISTRY.values() if s.family == 'tabular']


def meta_models() -> list[ModelSpec]:
    """All models with family='meta', in registration order."""
    return [s for s in _REGISTRY.values() if s.family == 'meta']


def sequential_models() -> list[ModelSpec]:
    """All models with family='sequential', in registration order."""
    return [s for s in _REGISTRY.values() if s.family == 'sequential']
