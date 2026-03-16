"""
Phase 5: Final model training on all xG era data (seasons 7-10).

After CV evaluation (ml/evaluate.py) confirms model quality, this script
trains each model on ALL available xG era data and serialises artefacts to
models/{position}_{model}.pkl and models/{position}_{model}_meta.json.

Models trained per position: baseline, ridge, lgbm.

Usage:
    python -m ml.train                         # train all positions and models
    python -m ml.train --position DEF          # single position
    python -m ml.train --position DEF --model lgbm   # single model
    python -m ml.train --eval-first            # run CV before training
    python -m ml.train --tune                  # Optuna tuning (requires optuna)
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features import build_feature_matrix, VALID_POSITIONS
from ml.evaluate import (
    get_feature_cols,
    stratified_impute,
    build_ridge,
    build_lgbm,
    run_cv,
    summarise_cv,
    beats_baseline,
    LGBM_BASE_PARAMS,
    LOGS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

_HERE    = Path(__file__).parent.parent
MODELS_DIR = _HERE / 'models'
VALID_MODELS = ('baseline', 'ridge', 'lgbm')


# ---------------------------------------------------------------------------
# Model training on full dataset
# ---------------------------------------------------------------------------

def train_baseline(
    df: pd.DataFrame,
    feat_cols: list[str],
    position: str,
    cv_metrics: pd.DataFrame | None = None,
) -> None:
    """
    Baseline 'model': just stores the training mean of pts_rolling_5gw as fallback.
    At inference, predict = pts_rolling_5gw; NaN -> stored mean.
    """
    fallback_mean = float(df['total_points'].mean())
    bundle = {
        'model_name':    'baseline',
        'position':      position,
        'feature_cols':  feat_cols,
        'fallback_mean': fallback_mean,
    }
    _save(position, 'baseline', bundle, df, cv_metrics)


def train_ridge(
    df: pd.DataFrame,
    feat_cols: list[str],
    position: str,
    cv_metrics: pd.DataFrame | None = None,
    alpha: float = 1.0,
) -> None:
    """
    Train Ridge on all xG era data. Imputation fit on full training set.
    """
    X = df[feat_cols]
    y = df['total_points']
    s = df['season_id']

    bundle = build_ridge(X, y, s, alpha=alpha)
    bundle['model_name'] = 'ridge'
    bundle['position']   = position

    _save(position, 'ridge', bundle, df, cv_metrics)


def train_lgbm(
    df: pd.DataFrame,
    feat_cols: list[str],
    position: str,
    cv_metrics: pd.DataFrame | None = None,
    tune: bool = False,
    extra_params: dict | None = None,
) -> None:
    """
    Train LightGBM on all xG era data.
    If tune=True, run Optuna on the last CV fold (fold 3) to find best hyperparams,
    then retrain on all data with those params.
    """
    X = df[feat_cols]
    y = df['total_points']

    tuned_params: dict | None = None
    if tune:
        log.info(f'[train_lgbm] {position}: running Optuna on fold 3 ...')
        from ml.evaluate import CV_FOLDS, split_fold, _tune_lgbm
        train_seasons, val_season = CV_FOLDS[-1]
        train_df, val_df = split_fold(df, train_seasons, val_season)
        X_tr = train_df[feat_cols]
        y_tr = train_df['total_points']
        X_v  = val_df[feat_cols]
        y_v  = val_df['total_points']
        tuned_params = _tune_lgbm(X_tr, y_tr, X_v, y_v, position)
        log.info(f'[train_lgbm] {position}: best tuned params {tuned_params}')

    combined_extra = {**(tuned_params or {}), **(extra_params or {})}
    bundle = build_lgbm(X, y, position, extra_params=combined_extra if combined_extra else None)
    bundle['model_name'] = 'lgbm'
    bundle['position']   = position

    _save(position, 'lgbm', bundle, df, cv_metrics)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _save(
    position: str,
    model_name: str,
    bundle: dict,
    df: pd.DataFrame,
    cv_metrics: pd.DataFrame | None,
) -> None:
    """Serialise model bundle (.pkl) and metadata (.json) to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f'{position}_{model_name}'

    # PKL: everything needed for inference
    pkl_path = MODELS_DIR / f'{stem}.pkl'
    joblib.dump(bundle, pkl_path)
    log.info(f'[save] {pkl_path}')

    # JSON: human-readable metadata
    meta: dict = {
        'position':      position,
        'model':         model_name,
        'feature_cols':  bundle.get('feature_cols', []),
        'train_seasons': sorted(df['season_id'].unique().tolist()),
        'n_train_rows':  len(df),
        'trained_at':    datetime.now(timezone.utc).isoformat(),
    }

    if cv_metrics is not None and not cv_metrics.empty:
        primary = cv_metrics[cv_metrics['model'] == model_name]
        if not primary.empty:
            meta['cv_mae_by_fold']    = primary['mae'].tolist()
            meta['cv_mae_mean']       = round(float(primary['mae'].mean()), 4)
            meta['cv_rmse_mean']      = round(float(primary['rmse'].mean()), 4)
            meta['cv_spearman_mean']  = round(float(primary['spearman'].mean()), 4)

    if model_name == 'lgbm' and 'params' in bundle:
        meta['lgbm_params'] = {
            k: v for k, v in bundle['params'].items()
            if k not in ('objective', 'metric', 'n_jobs', 'verbose', 'random_state')
        }

    json_path = MODELS_DIR / f'{stem}_meta.json'
    json_path.write_text(json.dumps(meta, indent=2))
    log.info(f'[save] {json_path}')


# ---------------------------------------------------------------------------
# CV-first helpers
# ---------------------------------------------------------------------------

def _load_cv_metrics(position: str) -> pd.DataFrame | None:
    """Load previously saved CV metrics for a position, if available."""
    path = LOGS_DIR / f'cv_metrics_{position}.csv'
    if path.exists():
        df = pd.read_csv(path)
        return df[df['model'].isin(['baseline', 'ridge', 'lgbm'])]
    return None


# ---------------------------------------------------------------------------
# Main training entry points
# ---------------------------------------------------------------------------

def train_position(
    position: str,
    models: tuple[str, ...] = VALID_MODELS,
    eval_first: bool = False,
    tune: bool = False,
) -> None:
    """Train all requested models for one position."""
    if eval_first:
        log.info(f'[eval_first] Running CV for {position} ...')
        metrics_df, _, _ = run_cv(position)
        summary       = summarise_cv(metrics_df)
        wins          = beats_baseline(summary)
        print(f'\n{position} CV summary:')
        print(summary.to_string())
        for m, w in wins.items():
            print(f'  {m} beats baseline: {"PASS" if w else "FAIL"}')
        cv_metrics = metrics_df
    else:
        cv_metrics = _load_cv_metrics(position)

    df        = build_feature_matrix(position)
    feat_cols = get_feature_cols(df)
    log.info(f'[train] {position}: {len(df):,} rows, {len(feat_cols)} features, '
             f'seasons {sorted(df["season_id"].unique().tolist())}')

    for model_name in models:
        log.info(f'[train] {position}/{model_name} ...')
        if model_name == 'baseline':
            train_baseline(df, feat_cols, position, cv_metrics)
        elif model_name == 'ridge':
            train_ridge(df, feat_cols, position, cv_metrics)
        elif model_name == 'lgbm':
            train_lgbm(df, feat_cols, position, cv_metrics, tune=tune)
        else:
            log.warning(f'Unknown model "{model_name}", skipping')


def train_all(
    eval_first: bool = False,
    tune: bool = False,
) -> None:
    """Train all Tier 1 models for all positions."""
    log.info('Training all positions: ' + ', '.join(VALID_POSITIONS))
    for position in VALID_POSITIONS:
        log.info(f'\n{"="*60}')
        log.info(f'Position: {position}')
        log.info('='*60)
        train_position(position, eval_first=eval_first, tune=tune)
    log.info(f'\n[done] All models serialised to {MODELS_DIR}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='FPL Phase 5 final model training')
    p.add_argument('--position', choices=list(VALID_POSITIONS),
                   help='Train for one position only (default: all)')
    p.add_argument('--model', choices=list(VALID_MODELS),
                   help='Train one model only (default: all)')
    p.add_argument('--eval-first', action='store_true',
                   help='Run CV evaluation before training')
    p.add_argument('--tune', action='store_true',
                   help='Optuna tuning for LightGBM (uses fold 3 for search)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    models = (args.model,) if args.model else VALID_MODELS

    if args.position:
        train_position(
            args.position,
            models=models,
            eval_first=args.eval_first,
            tune=args.tune,
        )
    else:
        train_all(eval_first=args.eval_first, tune=args.tune)
