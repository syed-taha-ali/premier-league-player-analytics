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

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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
    """
    Build a rolling-mean baseline bundle.
    Prediction = pts_rolling_5gw; falls back to training-fold mean when NaN.
    """
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
    """Return pts_rolling_5gw as the prediction, substituting the training mean for NaN rows."""
    fallback = bundle.get('fallback_mean', 1.0)
    return X['pts_rolling_5gw'].fillna(fallback).values


_XGI_COL = 'xgi_rolling_5gw'
_XGI_DROP_POSITIONS = frozenset({'MID', 'FWD'})


def _build_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    alpha: float = 1.0,
    **kwargs,
) -> dict:
    """
    Build a Ridge regression bundle with stratified imputation and StandardScaler.
    Drops xgi_rolling_5gw for MID and FWD to resolve xG/xA/xGI collinearity.
    """
    from ml.evaluate import build_ridge

    # xg + xa already captures xgi signal; drop collinear xgi for MID/FWD
    if position in _XGI_DROP_POSITIONS and _XGI_COL in X_train.columns:
        drop_cols = [_XGI_COL]
        X_train = X_train.drop(columns=drop_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=drop_cols, errors='ignore')

    return build_ridge(X_train, y_train, sid_train, X_val, sid_val, alpha=alpha)


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
    """Predict using the fitted Ridge bundle via the shared scaled-linear inference path."""
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
    """
    Build a LightGBM bundle using position-specific hyperparameters.
    Native NaN support — no imputation or scaling applied.
    If tune=True, runs Optuna on fold 3 to find optimal hyperparameters first.
    """
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
    """Predict using a fitted LightGBM model; NaN values are handled natively by LightGBM."""
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
    """Predict using stored home/away mean points; falls back to the training-fold mean."""
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
    """Build an ElasticNet bundle (alpha=1.0, l1_ratio=0.5) with imputation and scaling."""
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
    """
    Build a BayesianRidge bundle with imputation and scaling.
    Stores posterior predictive std in bundle['pred_std'] for uncertainty quantification.
    """
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
    """Build a Lasso bundle (alpha=1.0, max_iter=2000) with imputation and scaling."""
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
    """Add a ModelSpec to the module-level registry dict."""
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
    """Build an XGBoost bundle using tree_method='hist'; native NaN support, no imputation."""
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
    """Build a RandomForest bundle (n_estimators=200) with stratified mean imputation."""
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
    """Build an ExtraTrees bundle (n_estimators=200) with stratified mean imputation."""
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
    """Build a HistGradientBoosting bundle; native NaN support, no imputation or scaling."""
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
    """
    Build a CatBoost bundle (iterations=300, depth=6); native NaN support.
    Requires the catboost package — conditionally registered at module load.
    """
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
    # catboost is optional — register only if the package is installed
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
# Batch 3: poisson_glm, fdr_mean, last_season_avg
# ---------------------------------------------------------------------------

def _build_poisson_glm(
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
    Statsmodels GLM with Poisson family.

    total_points can be negative (red card = -1), so we shift the target by
    min_shift = max(0, 1 - y_min) so the minimum is 1. The shift is subtracted
    at inference time.
    """
    import statsmodels.api as sm
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

    min_shift = max(0.0, 1.0 - float(y_train.min()))
    y_shifted = y_train.values.astype(float) + min_shift

    X_tr_const = np.column_stack([np.ones(len(X_tr_f)), X_tr_f.values])
    model = sm.GLM(
        y_shifted, X_tr_const,
        family=sm.families.Poisson(),
    ).fit(disp=False)

    preds = None
    if X_v_f is not None:
        X_v_const = np.column_stack([np.ones(len(X_v_f)), X_v_f.values])
        preds = model.predict(X_v_const) - min_shift

    return {
        'model':        model,
        'season_means': season_means,
        'global_means': global_means,
        'min_shift':    min_shift,
        'feature_cols': list(X_train.columns),
        'preds':        preds,
    }


def _predict_poisson_glm(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Predict using the fitted Poisson GLM; applies imputation and reverses the target shift."""
    feat_cols    = bundle['feature_cols']
    season_means = bundle['season_means']
    global_means = bundle['global_means']
    model        = bundle['model']
    min_shift    = bundle['min_shift']

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

    X_const = np.column_stack([np.ones(len(X_filled)), X_filled])
    return model.predict(X_const) - min_shift


def _fdr_bin(rank: float) -> str:
    """Map opponent_season_rank to difficulty bin A/B/C."""
    if rank <= 6:
        return 'A'
    elif rank <= 14:
        return 'B'
    else:
        return 'C'


def _build_fdr_mean(
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
    Non-ML fixture-difficulty multiplier model.

    Bins opponent_season_rank into A (top 6), B (7-14), C (15-20).
    Multiplier for each bin = mean(actual_pts) / mean(pts_rolling_5gw) on
    training fold. Prediction = pts_rolling_5gw * multiplier[bin].
    """
    fallback = float(y_train.mean())
    rolling  = X_train['pts_rolling_5gw'].fillna(0.0)
    bins_tr  = X_train['opponent_season_rank'].fillna(10.0).map(_fdr_bin)

    multipliers: dict[str, float] = {}
    for b in ('A', 'B', 'C'):
        mask = bins_tr == b
        if mask.any():
            mean_rolling = float(rolling[mask].mean())
            multipliers[b] = (
                float(y_train[mask].mean()) / mean_rolling
                if mean_rolling > 0.0
                else 1.0
            )
        else:
            multipliers[b] = 1.0

    preds = None
    if X_val is not None:
        rolling_v = X_val['pts_rolling_5gw'].reset_index(drop=True)
        bins_v    = X_val['opponent_season_rank'].fillna(10.0).map(_fdr_bin).reset_index(drop=True)
        preds = np.array([
            rolling_v.iloc[i] * multipliers.get(bins_v.iloc[i], 1.0)
            if not pd.isna(rolling_v.iloc[i])
            else fallback
            for i in range(len(X_val))
        ])

    return {
        'multipliers': multipliers,
        'fallback':    fallback,
        'feature_cols': list(X_train.columns),
        'preds':        preds,
    }


def _predict_fdr_mean(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Scale pts_rolling_5gw by FDR-bin multipliers trained from CV fold data."""
    multipliers = bundle['multipliers']
    fallback    = bundle['fallback']
    rolling_v   = X['pts_rolling_5gw'].reset_index(drop=True)
    bins_v      = X['opponent_season_rank'].fillna(10.0).map(_fdr_bin).reset_index(drop=True)
    return np.array([
        rolling_v.iloc[i] * multipliers.get(bins_v.iloc[i], 1.0)
        if not pd.isna(rolling_v.iloc[i])
        else fallback
        for i in range(len(X))
    ])


def _build_last_season_avg(
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
    Cold-start model: GW1 rows use player's mean pts from the prior season
    (taken from the training fold). All other rows use pts_rolling_5gw.

    Requires _train_df and _val_df kwargs for player_code / gw / season_id context.
    Logs GW1-specific MAE as a supplementary diagnostic.
    """
    import logging
    log = logging.getLogger(__name__)

    _train_df       = kwargs.get('_train_df')
    _val_df         = kwargs.get('_val_df')
    global_fallback = float(y_train.mean())

    # Build lookup: (player_code, season_id) -> mean total_points on training fold
    player_prior_season: dict[tuple, float] = {}
    if _train_df is not None:
        grp = _train_df.groupby(['player_code', 'season_id'])['total_points'].mean()
        player_prior_season = {k: float(v) for k, v in grp.items()}

    preds = None
    if X_val is not None and _val_df is not None:
        rolling_v = X_val['pts_rolling_5gw'].reset_index(drop=True)
        gw_v      = _val_df['gw'].reset_index(drop=True)
        player_v  = _val_df['player_code'].reset_index(drop=True)
        season_v  = _val_df['season_id'].reset_index(drop=True)

        preds      = np.empty(len(X_val))
        gw1_idxs  = []
        for i in range(len(X_val)):
            if int(gw_v.iloc[i]) == 1:
                prior_key = (player_v.iloc[i], int(season_v.iloc[i]) - 1)
                preds[i]  = player_prior_season.get(prior_key, global_fallback)
                gw1_idxs.append(i)
            else:
                r        = rolling_v.iloc[i]
                preds[i] = r if not pd.isna(r) else global_fallback

        if gw1_idxs and y_val is not None:
            from sklearn.metrics import mean_absolute_error
            gw1_mae = mean_absolute_error(
                y_val.iloc[gw1_idxs].values, preds[gw1_idxs]
            )
            log.info(
                f'[last_season_avg] {position} GW1 MAE = {gw1_mae:.4f} '
                f'(n_gw1={len(gw1_idxs)})'
            )

    return {
        'player_prior_season': player_prior_season,
        'global_fallback':     global_fallback,
        'feature_cols':        list(X_train.columns),
        'preds':               preds,
    }


def _predict_last_season_avg(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Predict using player's prior-season average, falling back to rolling form."""
    _df            = kwargs.get('_df')
    player_prior   = bundle['player_prior_season']
    global_fallback = bundle['global_fallback']
    rolling_v      = X['pts_rolling_5gw'].reset_index(drop=True)

    n     = len(X)
    preds = np.empty(n)

    if _df is not None:
        gw_v     = _df['gw'].reset_index(drop=True)
        player_v = _df['player_code'].reset_index(drop=True)
        season_v = _df['season_id'].reset_index(drop=True)
        for i in range(n):
            if int(gw_v.iloc[i]) == 1:
                prior_key = (player_v.iloc[i], int(season_v.iloc[i]) - 1)
                preds[i]  = player_prior.get(prior_key, global_fallback)
            else:
                r        = rolling_v.iloc[i]
                preds[i] = r if not pd.isna(r) else global_fallback
    else:
        for i in range(n):
            r        = rolling_v.iloc[i]
            preds[i] = r if not pd.isna(r) else global_fallback

    return preds


# Batch 3 registrations --------------------------------------------------

_register(ModelSpec(
    name='poisson_glm',
    family='tabular',
    tier=2,
    requires_imputation=True,
    requires_scaling=False,
    build_fn=_build_poisson_glm,
    predict_fn=_predict_poisson_glm,
))

_register(ModelSpec(
    name='fdr_mean',
    family='tabular',
    tier=2,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_fdr_mean,
    predict_fn=_predict_fdr_mean,
))

_register(ModelSpec(
    name='last_season_avg',
    family='tabular',
    tier=2,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_last_season_avg,
    predict_fn=_predict_last_season_avg,
))


# ---------------------------------------------------------------------------
# Batch 4: mlp (tabular), simple_avg and stacking (meta)
# ---------------------------------------------------------------------------

def _build_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sid_train: pd.Series | None = None,
    sid_val: pd.Series | None = None,
    **kwargs,
) -> dict:
    """Fit a 2-layer MLP with stratified imputation + StandardScaler. GK uses a smaller network."""
    from sklearn.neural_network import MLPRegressor
    hidden = (32, 16) if position == 'GK' else (64, 32)
    estimator = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return _build_scaled_linear(
        estimator, X_train, y_train, position, X_val, y_val, sid_train, sid_val
    )


# ---------------------------------------------------------------------------
# Meta-model helpers
#
# Meta-model build_fn signature differs from tabular:
#     build_fn(dep_preds: dict, y_val: np.ndarray, position: str, **kwargs) -> bundle
#
#   dep_preds    : {model_name: np.ndarray} of val-fold predictions for each dep
#   y_val        : actual targets for this validation fold
#   _oof_records : list[dict] of previous-fold OOF entries passed by evaluate.py;
#                  each entry has {model_name: preds_array, 'y': actuals_array}
#
# Meta-model predict_fn signature is the same as tabular:
#     predict_fn(bundle, X, sid=None, **kwargs) -> np.ndarray
# X is the raw feature DataFrame (not used directly); base predictions are
# supplied via the _dep_preds kwarg by predict.py.
# ---------------------------------------------------------------------------

def _build_simple_avg(
    dep_preds: dict,
    y_val: np.ndarray,
    position: str,
    **kwargs,
) -> dict:
    """
    Uniform-weight average of base model predictions.
    No fitting required — weights are equal across all deps with non-None preds.
    """
    base_models = [m for m, p in dep_preds.items() if p is not None]
    if not base_models:
        return {
            'base_models': [],
            'weights':     np.array([]),
            'feature_cols': [],
            'preds':       np.zeros(len(y_val)),
        }
    preds_arr = np.column_stack([dep_preds[m] for m in base_models])
    weights   = np.ones(len(base_models)) / len(base_models)
    return {
        'base_models': base_models,
        'weights':     weights,
        'feature_cols': base_models,
        'preds':        preds_arr @ weights,
    }


def _predict_simple_avg(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Return the column-mean of all available base model predictions."""
    _dep_preds  = kwargs.get('_dep_preds', {})
    base_models = bundle['base_models']
    available   = [m for m in base_models if m in _dep_preds and _dep_preds[m] is not None]
    if not available:
        log.warning('[simple_avg] No dep_preds provided -- returning zeros')
        return np.zeros(len(X))
    preds_arr = np.column_stack([_dep_preds[m] for m in available])
    return preds_arr.mean(axis=1)


def _build_stacking(
    dep_preds: dict,
    y_val: np.ndarray,
    position: str,
    **kwargs,
) -> dict:
    """
    Ridge (alpha=0.5) meta-learner stacking.

    Fitted only when >= 2 folds of OOF predictions are available (fold 3 only).
    For folds 1-2, falls back to equal-weight average since there is no OOF
    training data yet. Single-fold evaluation is intentional (Tier 3 model).
    """
    from sklearn.linear_model import Ridge as _Ridge
    from sklearn.preprocessing import StandardScaler as _Scaler

    _oof_records = kwargs.get('_oof_records', [])
    base_models  = [m for m, p in dep_preds.items() if p is not None]

    def _fallback_bundle(preds: np.ndarray) -> dict:
        return {
            'meta_model':  None,
            'scaler':      None,
            'base_models': base_models,
            'feature_cols': base_models,
            'preds':        preds,
        }

    if not base_models:
        return _fallback_bundle(np.zeros(len(y_val)))

    X_meta_v = np.column_stack([dep_preds[m] for m in base_models])

    if len(_oof_records) < 2:
        return _fallback_bundle(X_meta_v.mean(axis=1))

    # Build OOF training matrix from previous folds
    oof_Xs, oof_ys = [], []
    for record in _oof_records:
        available = [m for m in base_models if record.get(m) is not None]
        if available:
            oof_Xs.append(np.column_stack([record[m] for m in available]))
            oof_ys.append(record['y'])

    if not oof_Xs:
        return _fallback_bundle(X_meta_v.mean(axis=1))

    X_meta_tr = np.vstack(oof_Xs)
    y_meta_tr = np.concatenate(oof_ys)

    scaler      = _Scaler()
    meta_model  = _Ridge(alpha=0.5, random_state=42)
    meta_model.fit(scaler.fit_transform(X_meta_tr), y_meta_tr)
    preds = meta_model.predict(scaler.transform(X_meta_v))

    return {
        'meta_model':  meta_model,
        'scaler':      scaler,
        'base_models': base_models,
        'feature_cols': base_models,
        'preds':        preds,
    }


def _predict_stacking(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Run base-model predictions through the fitted meta-model (Ridge stacker)."""
    _dep_preds  = kwargs.get('_dep_preds', {})
    meta_model  = bundle.get('meta_model')
    scaler      = bundle.get('scaler')
    base_models = bundle['base_models']

    available = [m for m in base_models if m in _dep_preds and _dep_preds[m] is not None]
    if not available:
        log.warning('[stacking] No dep_preds provided -- returning zeros')
        return np.zeros(len(X))

    preds_arr = np.column_stack([_dep_preds[m] for m in available])
    if meta_model is None or scaler is None:
        return preds_arr.mean(axis=1)
    return meta_model.predict(scaler.transform(preds_arr))


# Batch 4 registrations (inserted before Batch 5 below) -----------------

_register(ModelSpec(
    name='mlp',
    family='tabular',
    tier=2,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_build_mlp,
    predict_fn=_predict_scaled_linear,
))

_register(ModelSpec(
    name='simple_avg',
    family='meta',
    tier=2,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_simple_avg,
    predict_fn=_predict_simple_avg,
    deps=['ridge', 'xgb', 'elasticnet', 'lgbm'],
))

_register(ModelSpec(
    name='stacking',
    family='meta',
    tier=3,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_stacking,
    predict_fn=_predict_stacking,
    deps=['ridge', 'xgb', 'elasticnet', 'lgbm', 'random_forest'],
))


# ---------------------------------------------------------------------------
# Batch 5: minutes_model, component_model (decomposed), poly_ridge (tabular),
#          blending (meta)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared helpers for decomposed models
# ---------------------------------------------------------------------------

def _load_raw_gw_cols(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Load raw per-GW fact columns from fpl.db and left-join onto df.

    df must contain: player_code, season_id, gw, fixture_id.
    Returns df with the requested columns merged in.
    """
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent / 'db' / 'fpl.db'
    keys    = ['player_code', 'season_id', 'gw', 'fixture_id']
    col_str = ', '.join(keys + columns)
    conn    = sqlite3.connect(str(db_path))
    raw     = pd.read_sql_query(
        f'SELECT {col_str} FROM fact_gw_player WHERE mng_win IS NULL', conn
    )
    conn.close()
    for k in keys:
        if k in raw.columns and k in df.columns:
            raw[k] = raw[k].astype(df[k].dtype)
    return df.merge(raw, on=keys, how='left')


def _apply_stored_imputation(
    X: pd.DataFrame,
    feat_cols: list[str],
    season_means: pd.DataFrame,
    global_means: pd.Series,
    sid: pd.Series | None,
) -> np.ndarray:
    """Apply stored per-season mean imputation. Returns filled float array."""
    X_arr = X[feat_cols].reset_index(drop=True).values.astype(float)
    gm    = global_means[feat_cols].values

    if sid is not None:
        s_vals = sid.reset_index(drop=True).values
        fill   = np.empty_like(X_arr)
        for i, s in enumerate(s_vals):
            fill[i] = (
                season_means.loc[s, feat_cols].values
                if s in season_means.index else gm
            )
        fill = np.where(np.isnan(fill), gm, fill)
    else:
        fill = np.tile(gm, (len(X_arr), 1))

    nan_mask = np.isnan(X_arr)
    X_filled = np.where(nan_mask, fill, X_arr)
    return np.where(np.isnan(X_filled), gm, X_filled)


# ---------------------------------------------------------------------------
# minutes_model (decomposed_minutes)
# ---------------------------------------------------------------------------

_CLF_FEATURE_COLS = [
    'mins_rolling_3gw',
    'season_starts_rate_to_date',
    'value_lag1',
    'transfers_in_lag1',
]


def _build_minutes_model(
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
    Two-stage decomposed model.
    Stage 1 : LogisticRegression on rotation-signal features -> P(starts)
    Stage 2a: Ridge on full feature set for started rows -> E[pts | started]
    Stage 2b: Ridge on full feature set for benched rows -> E[pts | benched]
    Prediction: P(starts)*E[pts|started] + (1-P(starts))*E[pts|benched]

    Loads actual 'starts' column from fpl.db via _train_df context kwarg.
    Falls back to season_starts_rate_to_date >= 0.5 threshold if unavailable.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from ml.evaluate import stratified_impute

    _train_df = kwargs.get('_train_df')

    # Load starts target from fpl.db (training rows only — no leakage)
    if _train_df is not None:
        tr_aug   = _load_raw_gw_cols(
            _train_df[['player_code', 'season_id', 'gw', 'fixture_id']], ['starts']
        )
        y_starts = tr_aug['starts'].fillna(0).astype(int).values
    else:
        y_starts = (
            X_train['season_starts_rate_to_date'].fillna(0.5) >= 0.5
        ).astype(int).values

    starts_mask = y_starts == 1
    n_started   = int(starts_mask.sum())
    n_benched   = int((~starts_mask).sum())
    log.info(f'[minutes_model] {position}: n_started={n_started}, n_benched={n_benched}')

    # Single impute pass for the full feature set
    if X_val is not None and sid_val is not None:
        X_tr_f, X_v_f, season_means, global_means = stratified_impute(
            X_train, X_val, sid_train, sid_val
        )
    else:
        X_tr_f, _, season_means, global_means = stratified_impute(
            X_train, X_train, sid_train, sid_train
        )
        X_v_f = None

    feat_cols = list(X_train.columns)
    clf_cols  = [c for c in _CLF_FEATURE_COLS if c in feat_cols]
    clf_idx   = [feat_cols.index(c) for c in clf_cols]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr_f)

    # Stage 1: LogisticRegression on clf feature subset
    clf = LogisticRegression(random_state=42, max_iter=500, C=1.0)
    clf.fit(X_tr_s[:, clf_idx], y_starts)

    # Stage 2: Ridge sub-models; fall back to full data when split is too small
    _MIN        = 10
    reg_started = Ridge(alpha=1.0, random_state=42)
    reg_benched = Ridge(alpha=1.0, random_state=42)
    reg_started.fit(
        X_tr_s[starts_mask]  if n_started >= _MIN else X_tr_s,
        y_train.values[starts_mask] if n_started >= _MIN else y_train.values,
    )
    reg_benched.fit(
        X_tr_s[~starts_mask] if n_benched >= _MIN else X_tr_s,
        y_train.values[~starts_mask] if n_benched >= _MIN else y_train.values,
    )

    preds = None
    if X_v_f is not None:
        X_v_s   = scaler.transform(X_v_f)
        p_start = clf.predict_proba(X_v_s[:, clf_idx])[:, 1]
        preds   = (
            p_start * reg_started.predict(X_v_s)
            + (1 - p_start) * reg_benched.predict(X_v_s)
        )

    return {
        'clf':          clf,
        'clf_cols':     clf_cols,
        'clf_idx':      clf_idx,
        'reg_started':  reg_started,
        'reg_benched':  reg_benched,
        'scaler':       scaler,
        'season_means': season_means,
        'global_means': global_means,
        'feature_cols': feat_cols,
        'preds':        preds,
    }


def _predict_minutes_model(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Predict expected minutes using the Ridge minutes model bundle."""
    feat_cols    = bundle['feature_cols']
    season_means = bundle['season_means']
    global_means = bundle['global_means']
    scaler       = bundle['scaler']
    clf          = bundle['clf']
    clf_idx      = bundle['clf_idx']
    reg_started  = bundle['reg_started']
    reg_benched  = bundle['reg_benched']

    X_filled = _apply_stored_imputation(X, feat_cols, season_means, global_means, sid)
    X_s      = scaler.transform(X_filled)
    p_start  = clf.predict_proba(X_s[:, clf_idx])[:, 1]
    return (
        p_start * reg_started.predict(X_s)
        + (1 - p_start) * reg_benched.predict(X_s)
    )


# ---------------------------------------------------------------------------
# component_model (decomposed_components)
# ---------------------------------------------------------------------------

_SCORING_RULES: dict[str, dict[str, int]] = {
    'GK':  {'goals': 6, 'assists': 3, 'cs': 6, 'bonus': 1},
    'DEF': {'goals': 6, 'assists': 3, 'cs': 6, 'bonus': 1},
    'MID': {'goals': 5, 'assists': 3, 'cs': 1, 'bonus': 1},
    'FWD': {'goals': 4, 'assists': 3, 'cs': 0, 'bonus': 1},
}

_COMPONENT_DB_COLS: dict[str, str] = {
    'goals':   'goals_scored',
    'assists':  'assists',
    'cs':       'clean_sheets',
    'bonus':    'bonus',
}


def _build_component_model(
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
    One Ridge regression per FPL scoring component (goals, assists, clean_sheet, bonus).
    Prediction = sum(ridge_pred[component] * scoring_rule[component]).

    Loads component target columns from fpl.db via _train_df context kwarg.
    Components with scoring_rule == 0 for the position are skipped.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from ml.evaluate import stratified_impute

    _train_df = kwargs.get('_train_df')
    scoring   = _SCORING_RULES[position]

    # Load component columns from DB
    if _train_df is not None:
        tr_aug = _load_raw_gw_cols(
            _train_df[['player_code', 'season_id', 'gw', 'fixture_id']],
            list(_COMPONENT_DB_COLS.values()),
        )
    else:
        tr_aug = pd.DataFrame()

    # Shared imputation for all sub-models
    if X_val is not None and sid_val is not None:
        X_tr_f, X_v_f, season_means, global_means = stratified_impute(
            X_train, X_val, sid_train, sid_val
        )
    else:
        X_tr_f, _, season_means, global_means = stratified_impute(
            X_train, X_train, sid_train, sid_train
        )
        X_v_f = None

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_f)

    sub_models: dict[str, object] = {}
    for comp, db_col in _COMPONENT_DB_COLS.items():
        if scoring.get(comp, 0) == 0:
            sub_models[comp] = None
            continue
        y_comp = (
            tr_aug[db_col].fillna(0).values
            if db_col in tr_aug.columns
            else np.zeros(len(X_tr_s))
        )
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_tr_s, y_comp)
        sub_models[comp] = ridge

    preds = None
    if X_v_f is not None:
        X_v_s = scaler.transform(X_v_f)
        preds = np.zeros(len(X_v_s))
        for comp, ridge in sub_models.items():
            if ridge is not None:
                preds += ridge.predict(X_v_s) * scoring[comp]

    return {
        'models':        sub_models,
        'scaler':        scaler,
        'season_means':  season_means,
        'global_means':  global_means,
        'scoring_rules': scoring,
        'feature_cols':  list(X_train.columns),
        'preds':         preds,
    }


def _predict_component_model(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Predict using the component model (Ridge trained on per-action component targets)."""
    feat_cols    = bundle['feature_cols']
    season_means = bundle['season_means']
    global_means = bundle['global_means']
    scaler       = bundle['scaler']
    sub_models   = bundle['models']
    scoring      = bundle['scoring_rules']

    X_filled = _apply_stored_imputation(X, feat_cols, season_means, global_means, sid)
    X_s      = scaler.transform(X_filled)
    preds    = np.zeros(len(X_s))
    for comp, ridge in sub_models.items():
        if ridge is not None:
            preds += ridge.predict(X_s) * scoring[comp]
    return preds


# ---------------------------------------------------------------------------
# poly_ridge (Tier 3 standalone, tabular)
# ---------------------------------------------------------------------------

def _build_poly_ridge(
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
    Degree-2 pairwise interaction features + Ridge.
    Pipeline: impute -> scale -> PolynomialFeatures(interaction_only=True) -> Ridge.

    With n=20 features, interaction_only=True gives C(20,2)=190 cross-product
    terms plus the original 20, totalling 210 features fed to Ridge.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr_f)
    poly    = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_tr_p  = poly.fit_transform(X_tr_s)
    model   = Ridge(alpha=1.0, random_state=42)
    model.fit(X_tr_p, y_train.values)

    preds = None
    if X_v_f is not None:
        preds = model.predict(poly.transform(scaler.transform(X_v_f)))

    return {
        'model':        model,
        'scaler':       scaler,
        'poly':         poly,
        'season_means': season_means,
        'global_means': global_means,
        'feature_cols': list(X_train.columns),
        'preds':        preds,
    }


def _predict_poly_ridge(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
    """Predict using the degree-2 polynomial Ridge bundle (imputation + scale + poly transform)."""
    feat_cols    = bundle['feature_cols']
    season_means = bundle['season_means']
    global_means = bundle['global_means']
    scaler       = bundle['scaler']
    poly         = bundle['poly']
    model        = bundle['model']

    X_filled = _apply_stored_imputation(X, feat_cols, season_means, global_means, sid)
    return model.predict(poly.transform(scaler.transform(X_filled)))


# ---------------------------------------------------------------------------
# blending (Tier 3, meta)
#
# Same OOF accumulation mechanism as stacking; differentiated by deps
# (top-performing models rather than diverse ensemble) and an ElasticNet
# meta-learner instead of Ridge.
# ---------------------------------------------------------------------------

def _build_blending(
    dep_preds: dict,
    y_val: np.ndarray,
    position: str,
    **kwargs,
) -> dict:
    """
    Blending meta-learner using ElasticNet (l1_ratio=0.5, alpha=0.5).
    Fitted on fold 1+2 OOF val predictions (_oof_records), evaluated on fold 3.
    Single-fold evaluation is intentional for this Tier 3 model.
    """
    from sklearn.linear_model import ElasticNet as _EN
    from sklearn.preprocessing import StandardScaler as _Scaler

    _oof_records = kwargs.get('_oof_records', [])
    base_models  = [m for m, p in dep_preds.items() if p is not None]

    def _fallback(preds: np.ndarray) -> dict:
        return {
            'meta_model':  None,
            'scaler':      None,
            'base_models': base_models,
            'feature_cols': base_models,
            'preds':        preds,
        }

    if not base_models:
        return _fallback(np.zeros(len(y_val)))

    X_meta_v = np.column_stack([dep_preds[m] for m in base_models])

    if len(_oof_records) < 2:
        return _fallback(X_meta_v.mean(axis=1))

    oof_Xs, oof_ys = [], []
    for record in _oof_records:
        available = [m for m in base_models if record.get(m) is not None]
        if available:
            oof_Xs.append(np.column_stack([record[m] for m in available]))
            oof_ys.append(record['y'])

    if not oof_Xs:
        return _fallback(X_meta_v.mean(axis=1))

    X_meta_tr = np.vstack(oof_Xs)
    y_meta_tr = np.concatenate(oof_ys)

    scaler     = _Scaler()
    meta_model = _EN(l1_ratio=0.5, alpha=0.5, max_iter=2000, random_state=42)
    meta_model.fit(scaler.fit_transform(X_meta_tr), y_meta_tr)
    preds = meta_model.predict(scaler.transform(X_meta_v))

    return {
        'meta_model':  meta_model,
        'scaler':      scaler,
        'base_models': base_models,
        'feature_cols': base_models,
        'preds':        preds,
    }


# Batch 5 registrations --------------------------------------------------

_register(ModelSpec(
    name='minutes_model',
    family='decomposed',
    tier=2,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_build_minutes_model,
    predict_fn=_predict_minutes_model,
))

_register(ModelSpec(
    name='component_model',
    family='decomposed',
    tier=3,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_build_component_model,
    predict_fn=_predict_component_model,
))

_register(ModelSpec(
    name='poly_ridge',
    family='tabular',
    tier=3,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_build_poly_ridge,
    predict_fn=_predict_poly_ridge,
))

_register(ModelSpec(
    name='blending',
    family='meta',
    tier=3,
    requires_imputation=False,
    requires_scaling=False,
    build_fn=_build_blending,
    predict_fn=_predict_stacking,  # same inference mechanism as stacking
    deps=['ridge', 'bayesian_ridge', 'poisson_glm', 'mlp'],
))


# ---------------------------------------------------------------------------
# Batch 6 — Sequential model stubs (LSTM, GRU)
#
# These are registered so that get_registry() / sequential_models() work,
# but build_fn / predict_fn are NOT called by the main tabular pipeline.
# All training and evaluation is handled by ml/evaluate_sequential.py.
# ---------------------------------------------------------------------------

def _seq_build_stub(X_train, y_train, position, **kwargs):
    """Raise NotImplementedError — sequential models use ml/evaluate_sequential.py."""
    raise NotImplementedError(
        'Sequential models must be trained via ml/evaluate_sequential.py, '
        'not through the main tabular train pipeline.'
    )


def _seq_predict_stub(bundle, X, **kwargs):
    """Raise NotImplementedError — sequential models use ml/evaluate_sequential.py."""
    raise NotImplementedError(
        'Sequential models are not supported by the tabular predict pipeline. '
        'Use ml/evaluate_sequential.py instead.'
    )


_register(ModelSpec(
    name='lstm',
    family='sequential',
    tier=3,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_seq_build_stub,
    predict_fn=_seq_predict_stub,
))

_register(ModelSpec(
    name='gru',
    family='sequential',
    tier=3,
    requires_imputation=True,
    requires_scaling=True,
    build_fn=_seq_build_stub,
    predict_fn=_seq_predict_stub,
))


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
    """All models with family='tabular' or 'decomposed', in registration order."""
    return [s for s in _REGISTRY.values() if s.family in ('tabular', 'decomposed')]


def meta_models() -> list[ModelSpec]:
    """All models with family='meta', in registration order."""
    return [s for s in _REGISTRY.values() if s.family == 'meta']


def sequential_models() -> list[ModelSpec]:
    """All models with family='sequential', in registration order."""
    return [s for s in _REGISTRY.values() if s.family == 'sequential']
