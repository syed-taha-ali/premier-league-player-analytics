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


def _predict_ridge(
    bundle: dict,
    X: pd.DataFrame,
    sid: pd.Series | None = None,
    **kwargs,
) -> np.ndarray:
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
