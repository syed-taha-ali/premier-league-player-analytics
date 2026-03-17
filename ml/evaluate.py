"""
Phase 5/6: Expanding-window CV evaluation for FPL GW-level prediction models.

CV folds (xG era, seasons 7-10):
    Fold 1: train season  7,       validate season  8   (2022-23 -> 2023-24)
    Fold 2: train seasons 7-8,     validate season  9   (2022-24 -> 2024-25)
    Fold 3: train seasons 7-8-9,   validate season 10   (2022-25 -> 2025-26)

Models are defined in ml/models.py (registry). The CV loop iterates over all
registered tabular models (Pass 1) and all registered meta-models (Pass 2).

Usage:
    python -m ml.evaluate                     # all positions, default hyperparams
    python -m ml.evaluate --position GK       # single position
    python -m ml.evaluate --tune              # Optuna search for LightGBM (requires optuna)
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from ml.features import build_feature_matrix, CONTEXT_COLS, TARGET_COL, VALID_POSITIONS
from ml.models import tabular_models, meta_models, get_registry

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.parent
LOGS_DIR         = _HERE / 'logs' / 'training'
OUTPUTS_MODELS   = _HERE / 'outputs' / 'models'
MODELS_DIR       = _HERE / 'models'

# ---------------------------------------------------------------------------
# CV fold definitions
# ---------------------------------------------------------------------------
# Each entry: (train_season_ids, val_season_id)
CV_FOLDS = [
    ([7],       8),
    ([7, 8],    9),
    ([7, 8, 9], 10),
]

FOLD_LABELS = {
    1: 'train 2022-23 -> val 2023-24',
    2: 'train 2022-24 -> val 2024-25',
    3: 'train 2022-25 -> val 2025-26',
}

# ---------------------------------------------------------------------------
# LightGBM hyperparameters (project_plan.md §5.3)
# ---------------------------------------------------------------------------
LGBM_BASE_PARAMS = {
    'GK':  dict(num_leaves=15, min_child_samples=30, learning_rate=0.05, n_estimators=200),
    'DEF': dict(num_leaves=31, min_child_samples=20, learning_rate=0.05, n_estimators=300),
    'MID': dict(num_leaves=31, min_child_samples=20, learning_rate=0.05, n_estimators=300),
    'FWD': dict(num_leaves=31, min_child_samples=20, learning_rate=0.05, n_estimators=300),
}

_LGBM_COMMON = dict(
    objective='regression',
    metric='mae',
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

TOP_N = 10  # top-N precision metric

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return predictive feature columns (everything except context + target)."""
    exclude = set(CONTEXT_COLS) | {TARGET_COL}
    return [c for c in df.columns if c not in exclude]


def split_fold(
    df: pd.DataFrame, train_seasons: list[int], val_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df['season_id'].isin(train_seasons)].copy()
    val   = df[df['season_id'] == val_season].copy()
    return train, val


# ---------------------------------------------------------------------------
# Stratified mean imputation (Ridge only)
# project_plan.md §5.3: impute within training fold, stratified by season_id.
# Never fit the imputer on the full dataset or the validation fold.
# ---------------------------------------------------------------------------

def stratified_impute(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    sid_train: pd.Series,
    sid_val: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Fill NaN using per-season means computed from the training fold only.

    Returns
    -------
    X_train_filled, X_val_filled, season_means, global_means
    season_means and global_means are saved alongside the model for inference.
    """
    feat_cols = list(X_train.columns)

    X_tr = X_train.reset_index(drop=True).values.astype(float)
    X_v  = X_val.reset_index(drop=True).values.astype(float)
    s_tr = sid_train.reset_index(drop=True).values
    s_v  = sid_val.reset_index(drop=True).values

    # Season-level means from training fold
    seasons = np.unique(s_tr)
    season_means_dict: dict[int, np.ndarray] = {}
    for s in seasons:
        mask = s_tr == s
        season_means_dict[s] = np.nanmean(X_tr[mask], axis=0)

    global_means_arr = np.nanmean(X_tr, axis=0)
    # Handle features that are NaN in all rows (fill with 0)
    global_means_arr = np.where(np.isnan(global_means_arr), 0.0, global_means_arr)

    def _fill(arr: np.ndarray, sids: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        nan_mask = np.isnan(arr)
        if not nan_mask.any():
            return arr
        fill_matrix = np.empty_like(arr)
        for i, s in enumerate(sids):
            fill_matrix[i] = season_means_dict.get(s, global_means_arr)
        # Any season_mean entry itself NaN (e.g., all-NaN feature in that season) -> global
        fill_matrix = np.where(np.isnan(fill_matrix), global_means_arr, fill_matrix)
        arr = np.where(nan_mask, fill_matrix, arr)
        # Final fallback (should not trigger)
        arr = np.where(np.isnan(arr), global_means_arr, arr)
        return arr

    X_tr_filled = pd.DataFrame(_fill(X_tr, s_tr), columns=feat_cols)
    X_v_filled  = pd.DataFrame(_fill(X_v,  s_v),  columns=feat_cols)

    season_means = pd.DataFrame.from_dict(
        season_means_dict, orient='index', columns=feat_cols
    )
    global_means = pd.Series(global_means_arr, index=feat_cols)

    return X_tr_filled, X_v_filled, season_means, global_means


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def baseline_predict(
    X_val: pd.DataFrame, fallback_mean: float
) -> np.ndarray:
    """Predict using pts_rolling_5gw; NaN filled with training mean."""
    preds = X_val['pts_rolling_5gw'].copy()
    return preds.fillna(fallback_mean).values


def build_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sid_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    sid_val: pd.Series | None = None,
    alpha: float = 1.0,
) -> dict:
    """
    Fit Ridge with stratified imputation + StandardScaler.

    Returns a bundle dict containing the model artefacts needed for inference.
    If X_val is provided, also returns predictions on the validation set.
    """
    # Impute (or just impute training if no validation)
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

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_tr_s, y_train.values)

    preds = None
    if X_v_f is not None:
        X_v_s = scaler.transform(X_v_f)
        preds = model.predict(X_v_s)

    return {
        'model':        model,
        'scaler':       scaler,
        'season_means': season_means,
        'global_means': global_means,
        'feature_cols': list(X_train.columns),
        'preds':        preds,
    }


def build_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    position: str,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    tune: bool = False,
    extra_params: dict | None = None,
) -> dict:
    """
    Fit LightGBM. Native NaN handling -- no imputation.

    If tune=True and optuna is available, run Bayesian hyperparameter search
    using X_val/y_val as the evaluation set (project_plan.md §5.3).
    """
    params = {**LGBM_BASE_PARAMS[position], **_LGBM_COMMON}
    if extra_params:
        params.update(extra_params)

    if tune and HAS_OPTUNA and X_val is not None and y_val is not None:
        best = _tune_lgbm(X_train, y_train, X_val, y_val, position)
        params.update(best)
        log.info(f"[lgbm_tune] {position}: best params {best}")

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val) if X_val is not None else None

    return {
        'model':        model,
        'feature_cols': list(X_train.columns),
        'params':       params,
        'preds':        preds,
    }


def _tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    position: str,
    n_trials: int = 40,
) -> dict:
    """Optuna TPE search for LightGBM. Returns best hyperparameter dict."""

    def objective(trial: optuna.Trial) -> float:
        p = {
            **_LGBM_COMMON,
            'num_leaves':        trial.suggest_int('num_leaves', 8, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
            'learning_rate':     trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500, step=50),
            'feature_fraction':  trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq':      1,
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        }
        m = lgb.LGBMRegressor(**p)
        m.fit(X_train, y_train)
        return mean_absolute_error(y_val, m.predict(X_val))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"[tune] {position}: best MAE={study.best_value:.4f}")
    return study.best_params


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gw_season_ids: np.ndarray | None = None,
    top_n: int = TOP_N,
) -> dict:
    """
    Compute MAE, RMSE, R², Spearman rho, and Top-N per-GW precision.

    gw_season_ids: 1-D array of (season_id * 1000 + gw) for per-GW grouping.
    If None, Top-N is computed over the whole set.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)

    if gw_season_ids is not None:
        precisions = []
        for gw_key in np.unique(gw_season_ids):
            m = gw_season_ids == gw_key
            if m.sum() <= top_n:
                continue
            yt = y_true[m]
            yp = y_pred[m]
            true_top = set(np.argsort(yt)[-top_n:])
            pred_top = set(np.argsort(yp)[-top_n:])
            precisions.append(len(true_top & pred_top) / top_n)
        top_n_prec = float(np.mean(precisions)) if precisions else float('nan')
    else:
        true_top = set(np.argsort(y_true)[-top_n:])
        pred_top = set(np.argsort(y_pred)[-top_n:])
        top_n_prec = len(true_top & pred_top) / top_n

    return {
        'mae':      round(float(mae),       4),
        'rmse':     round(float(rmse),      4),
        'r2':       round(float(r2),        4),
        'spearman': round(float(rho),       4),
        f'top{top_n}_prec': round(top_n_prec, 4),
    }


def compute_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Metrics for home/away and opponent-tier subgroups.
    Uses was_home and opponent_season_rank from val_df.
    """
    gw_keys = (val_df['season_id'].values * 1000 + val_df['gw'].values
               if 'gw' in val_df.columns else None)
    results = {}

    if 'was_home' in val_df.columns:
        for label, mask in [('home', val_df['was_home'] == 1),
                             ('away', val_df['was_home'] == 0)]:
            m = mask.values
            if m.sum() >= 10:
                gk = gw_keys[m] if gw_keys is not None else None
                results[label] = compute_metrics(y_true[m], y_pred[m], gk)

    if 'opponent_season_rank' in val_df.columns:
        for label, mask in [('vs_top6', val_df['opponent_season_rank'] <= 6),
                             ('vs_rest', val_df['opponent_season_rank'] > 6)]:
            m = mask.values
            if m.sum() >= 10:
                gk = gw_keys[m] if gw_keys is not None else None
                results[label] = compute_metrics(y_true[m], y_pred[m], gk)

    return results


# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------

def run_cv(
    position: str,
    tune_lgbm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run 3-fold expanding-window CV for one position.

    Returns
    -------
    metrics_df   -- one row per (fold, model[_subgroup])
    preds_df     -- per-row predictions across all folds
    last_models  -- models trained on fold 3 (for SHAP / serialisation)
    """
    df = build_feature_matrix(position)
    feat_cols = get_feature_cols(df)
    log.info(f"[cv] {position}: {len(df):,} rows, {len(feat_cols)} features")

    records     = []
    all_preds   = []
    last_models: dict  = {}
    oof_records: list[dict] = []   # accumulates tabular val predictions across folds for stacking

    for fold_idx, (train_seasons, val_season) in enumerate(CV_FOLDS, start=1):
        train_df, val_df = split_fold(df, train_seasons, val_season)
        if train_df.empty or val_df.empty:
            log.warning(f"[cv] {position} fold {fold_idx}: empty split, skipping")
            continue

        X_train = train_df[feat_cols].reset_index(drop=True)
        y_train = train_df[TARGET_COL].reset_index(drop=True)
        X_val   = val_df[feat_cols].reset_index(drop=True)
        y_val   = val_df[TARGET_COL].reset_index(drop=True)
        sid_tr  = train_df['season_id'].reset_index(drop=True)
        sid_v   = val_df['season_id'].reset_index(drop=True)
        gw_keys = (val_df['season_id'].values * 1000 + val_df['gw'].values)

        # ---- Pass 1: tabular models ----
        fold_bundles: dict[str, dict] = {}
        for spec in tabular_models():
            bundle = spec.build_fn(
                X_train, y_train, position,
                X_val=X_val, y_val=y_val,
                sid_train=sid_tr, sid_val=sid_v,
                tune=tune_lgbm,
                _train_df=train_df,
                _val_df=val_df,
            )
            preds = bundle['preds']
            _record_metrics(records, fold_idx, spec.name, y_val.values, preds, gw_keys, val_df)
            fold_bundles[spec.name] = bundle

        # Collect tabular val predictions for stacking OOF (used in later folds)
        oof_entry: dict = {'y': y_val.values}
        for _oname, _ob in fold_bundles.items():
            if _ob.get('preds') is not None:
                oof_entry[_oname] = _ob['preds']

        # ---- Pass 2: meta-models (fit on Pass 1 val predictions) ----
        for spec in meta_models():
            if not all(d in fold_bundles for d in spec.deps):
                log.warning(f"[cv] Skipping meta-model {spec.name}: deps not satisfied")
                continue
            dep_preds = {d: fold_bundles[d]['preds'] for d in spec.deps}
            bundle = spec.build_fn(
                dep_preds, y_val.values, position,
                _oof_records=list(oof_records),
            )
            preds = bundle['preds']
            _record_metrics(records, fold_idx, spec.name, y_val.values, preds, gw_keys, val_df)
            fold_bundles[spec.name] = bundle

        # Append current fold after Pass 2 so stacking uses only previous folds
        oof_records.append(oof_entry)

        # ---- Collect predictions ----
        fold_pred = val_df[CONTEXT_COLS + [TARGET_COL]].copy().reset_index(drop=True)
        fold_pred['fold'] = fold_idx
        for name, bundle in fold_bundles.items():
            if bundle.get('preds') is not None:
                fold_pred[f'pred_{name}'] = bundle['preds']
            if bundle.get('pred_std') is not None:
                fold_pred[f'pred_std_{name}'] = bundle['pred_std']
        all_preds.append(fold_pred)

        # Keep fold 3 models for diagnostics / final serialisation reference
        if fold_idx == len(CV_FOLDS):
            for name, bundle in fold_bundles.items():
                last_models[name] = bundle
            last_models['feat_cols'] = feat_cols
            last_models['X_val']     = X_val
            last_models['y_val']     = y_val

        # Log summary
        registered_names = list(get_registry().keys())
        fold_m = {r['model']: r['mae']
                  for r in records
                  if r['fold'] == fold_idx and r['model'] in registered_names}
        mae_str = '  '.join(
            f'{n}={fold_m.get(n, float("nan")):.3f}' for n in registered_names
        )
        log.info(
            f"[cv] {position} fold {fold_idx} ({FOLD_LABELS[fold_idx]}): "
            f"n_train={len(train_df):,} n_val={len(val_df):,} | MAE  {mae_str}"
        )

    metrics_df = pd.DataFrame(records)
    preds_df   = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return metrics_df, preds_df, last_models


def _record_metrics(
    records: list,
    fold_idx: int,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gw_keys: np.ndarray,
    val_df: pd.DataFrame,
) -> None:
    """Append overall + stratified metric dicts to records list."""
    m = compute_metrics(y_true, y_pred, gw_keys)
    records.append({'fold': fold_idx, 'model': model_name, **m})

    strat = compute_stratified_metrics(y_true, y_pred, val_df.reset_index(drop=True))
    for subgroup, sm in strat.items():
        records.append({'fold': fold_idx, 'model': f'{model_name}_{subgroup}', **sm})


# ---------------------------------------------------------------------------
# Metrics summary helpers
# ---------------------------------------------------------------------------

def summarise_cv(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean CV metrics across folds for all registered models.
    Returns DataFrame indexed by model name with mean MAE, RMSE, R², Spearman, TopN.
    """
    model_names = list(get_registry().keys())
    primary = metrics_df[metrics_df['model'].isin(model_names)]
    return (
        primary
        .groupby('model')[['mae', 'rmse', 'r2', 'spearman', f'top{TOP_N}_prec']]
        .mean()
        .round(4)
        .reindex(model_names)
    )


def beats_baseline(summary: pd.DataFrame) -> dict[str, bool]:
    """
    Check whether each non-baseline model beats the baseline on >= 2 of 3 primary
    metrics (MAE lower, RMSE lower, Spearman higher). project_plan.md §6.5.
    """
    results = {}
    if 'baseline' not in summary.index:
        return results
    bl = summary.loc['baseline']
    non_baseline = [n for n in get_registry() if n != 'baseline']
    for model in non_baseline:
        if model not in summary.index:
            results[model] = False
            continue
        m = summary.loc[model]
        wins = sum([
            m['mae']      < bl['mae'],
            m['rmse']     < bl['rmse'],
            m['spearman'] > bl['spearman'],
        ])
        results[model] = wins >= 2
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_calibration(
    preds_df: pd.DataFrame,
    position: str,
    n_bins: int = 10,
) -> None:
    """
    Calibration plot: mean predicted vs mean actual in quantile bins.
    Pooled across all CV folds.
    """
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
    _default_colors = ['#888888', '#2196F3', '#E53935', '#4CAF50', '#FF9800', '#9C27B0']
    registered_names = list(get_registry().keys())
    model_cols = [f'pred_{n}' for n in registered_names if f'pred_{n}' in preds_df.columns]
    model_labels = {f'pred_{n}': n.capitalize() for n in registered_names}
    colors = {
        f'pred_{n}': _default_colors[i % len(_default_colors)]
        for i, n in enumerate(registered_names)
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    y_true = preds_df[TARGET_COL].values

    for col in model_cols:
        y_pred = preds_df[col].values
        # Quantile bins on predictions
        try:
            bins = pd.qcut(y_pred, q=n_bins, duplicates='drop')
        except ValueError:
            continue
        bin_df = pd.DataFrame({'actual': y_true, 'pred': y_pred, 'bin': bins})
        agg = bin_df.groupby('bin', observed=True).agg(
            mean_pred=('pred', 'mean'),
            mean_actual=('actual', 'mean'),
        ).sort_values('mean_pred')
        ax.plot(
            agg['mean_pred'], agg['mean_actual'],
            marker='o', label=model_labels[col], color=colors[col], linewidth=1.8,
        )

    lo = min(y_true.min(), preds_df[model_cols].values.min()) - 0.5
    hi = max(y_true.max(), preds_df[model_cols].values.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.5, label='Perfect')
    ax.set_xlabel('Mean predicted points')
    ax.set_ylabel('Mean actual points')
    ax.set_title(f'{position} — Calibration (pooled CV folds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUTS_MODELS / f'calibration_{position}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"[plot] Saved {out}")


def plot_shap(
    lgbm_bundle: dict,
    position: str,
    max_display: int = 15,
) -> None:
    """SHAP mean |value| bar chart for the LightGBM model from the last CV fold."""
    if not HAS_SHAP:
        log.warning('[shap] shap not installed -- skipping SHAP plot')
        return

    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
    model     = lgbm_bundle['model']
    X_val     = lgbm_bundle.get('X_val')
    feat_cols = lgbm_bundle['feature_cols']

    if X_val is None:
        log.warning('[shap] No validation data in bundle -- skipping SHAP plot')
        return

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_val)

    mean_abs   = np.abs(shap_vals).mean(axis=0)
    order      = np.argsort(mean_abs)[::-1][:max_display]
    feat_names = np.array(feat_cols)[order]
    vals       = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.4 + 1)))
    y_pos = np.arange(len(order))
    ax.barh(y_pos, vals[::-1], color='#E53935', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'{position} LightGBM — Feature Importance (SHAP, fold 3 val)')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    out = OUTPUTS_MODELS / f'shap_{position}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"[plot] Saved {out}")


def plot_metrics_by_fold(
    metrics_df: pd.DataFrame,
    position: str,
) -> None:
    """MAE per fold per model — shows stability across folds."""
    OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
    _default_colors = ['#888888', '#2196F3', '#E53935', '#4CAF50', '#FF9800', '#9C27B0']
    registered_names = list(get_registry().keys())
    color_map = {
        n: _default_colors[i % len(_default_colors)]
        for i, n in enumerate(registered_names)
    }
    primary = metrics_df[metrics_df['model'].isin(registered_names)]
    if primary.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, grp in primary.groupby('model'):
        ax.plot(grp['fold'], grp['mae'], marker='o',
                label=model_name, color=color_map.get(model_name, 'black'), linewidth=1.8)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([FOLD_LABELS[i] for i in [1, 2, 3]], fontsize=7)
    ax.set_ylabel('MAE (points)')
    ax.set_title(f'{position} — MAE by CV fold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUTS_MODELS / f'mae_by_fold_{position}.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_cv_results(
    metrics_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    position: str,
    summary: pd.DataFrame,
    baseline_wins: dict,
) -> None:
    """Write metrics CSV, predictions parquet, and a markdown summary."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(LOGS_DIR / f'cv_metrics_{position}.csv', index=False)
    preds_df.to_parquet(LOGS_DIR / f'cv_preds_{position}.parquet', index=False)

    lines = [
        f'# CV Results — {position}\n',
        '## Mean metrics across 3 folds\n',
        summary.to_markdown(),
        '\n',
        '## Beats rolling-mean baseline on >= 2 of 3 metrics (MAE, RMSE, Spearman)\n',
    ]
    for model, beats in baseline_wins.items():
        status = 'YES' if beats else 'NO'
        lines.append(f'- {model}: {status}')

    model_names = list(get_registry().keys())
    lines += [
        '\n',
        '## Per-fold metrics (primary models)\n',
        metrics_df[metrics_df['model'].isin(model_names)].to_markdown(index=False),
    ]

    report_path = LOGS_DIR / f'cv_report_{position}.md'
    report_path.write_text('\n'.join(lines))
    log.info(f"[save] CV report -> {report_path}")


# ---------------------------------------------------------------------------
# Run all positions
# ---------------------------------------------------------------------------

def run_all_positions(tune_lgbm: bool = False) -> pd.DataFrame:
    """Run CV for all positions, save results, return combined metrics DataFrame."""
    all_metrics = []

    for position in VALID_POSITIONS:
        log.info(f"\n{'='*60}")
        log.info(f"Position: {position}")
        log.info('='*60)

        metrics_df, preds_df, last_models = run_cv(position, tune_lgbm=tune_lgbm)

        summary       = summarise_cv(metrics_df)
        baseline_wins = beats_baseline(summary)

        print(f'\n--- {position} mean CV metrics ---')
        print(summary.to_string())
        for model, beats in baseline_wins.items():
            status = 'PASS' if beats else 'FAIL'
            print(f'  {model} beats baseline: {status}')

        save_cv_results(metrics_df, preds_df, position, summary, baseline_wins)
        plot_calibration(preds_df, position)
        plot_metrics_by_fold(metrics_df, position)

        if last_models and HAS_SHAP:
            lgbm_bundle          = last_models['lgbm']
            lgbm_bundle['X_val'] = last_models.get('X_val')
            plot_shap(lgbm_bundle, position)

        metrics_df['position'] = position
        all_metrics.append(metrics_df)

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(LOGS_DIR / 'cv_metrics_all.csv', index=False)
    log.info(f'\n[done] CV complete. Results in {LOGS_DIR}')
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='FPL Phase 5/6 CV evaluation')
    p.add_argument('--position', choices=list(VALID_POSITIONS),
                   help='Run for one position only (default: all)')
    p.add_argument('--tune', action='store_true',
                   help='Run Optuna hyperparameter search for LightGBM')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if not HAS_OPTUNA and args.tune:
        log.warning('--tune requested but optuna is not installed; using default params')

    if args.position:
        log.info(f'Running CV for {args.position}')
        metrics_df, preds_df, last_models = run_cv(args.position, tune_lgbm=args.tune)
        summary       = summarise_cv(metrics_df)
        baseline_wins = beats_baseline(summary)
        print(f'\n--- {args.position} mean CV metrics ---')
        print(summary.to_string())
        for model, beats in baseline_wins.items():
            print(f'  {model} beats baseline: {"PASS" if beats else "FAIL"}')
        save_cv_results(metrics_df, preds_df, args.position, summary, baseline_wins)
        plot_calibration(preds_df, args.position)
        plot_metrics_by_fold(metrics_df, args.position)
        if last_models and HAS_SHAP:
            lgbm_bundle          = last_models['lgbm']
            lgbm_bundle['X_val'] = last_models.get('X_val')
            plot_shap(lgbm_bundle, args.position)
    else:
        run_all_positions(tune_lgbm=args.tune)
