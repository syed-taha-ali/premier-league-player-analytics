"""
Phase 5: Final model training on all xG era data (seasons 7-10).

After CV evaluation (ml/evaluate.py) confirms model quality, this script
trains each model on ALL available xG era data and serialises artefacts to
models/{position}_{model}.pkl and models/{position}_{model}_meta.json.

Tabular / decomposed models are trained on the full feature matrix.
Meta-models (simple_avg, stacking, blending) are trained on the OOF
predictions from logs/training/cv_preds_{position}.parquet, which must
already exist (run ml.evaluate first).

Usage:
    python -m ml.train                         # train all tabular+decomposed models
    python -m ml.train --meta                  # train meta-models from OOF parquets
    python -m ml.train --all                   # train everything (tabular + meta)
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
    run_cv,
    summarise_cv,
    beats_baseline,
    LOGS_DIR,
)
from ml.models import get_registry, get_model, tabular_models, meta_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

_HERE      = Path(__file__).parent.parent
MODELS_DIR = _HERE / 'models'


# ---------------------------------------------------------------------------
# Generic tabular model trainer
# ---------------------------------------------------------------------------

def _train_tabular(
    spec,
    df: pd.DataFrame,
    feat_cols: list[str],
    position: str,
    cv_metrics: pd.DataFrame | None = None,
    tune: bool = False,
) -> None:
    """
    Build a tabular model on the full dataset and serialise the bundle.

    For lgbm with tune=True, runs Optuna on the last CV fold to find best
    hyperparameters, then retrains on all data with those params.
    """
    X = df[feat_cols]
    y = df['total_points']
    s = df['season_id']

    extra_params: dict | None = None
    if spec.name == 'lgbm' and tune:
        log.info(f'[train] {position}/lgbm: running Optuna on fold 3 ...')
        from ml.evaluate import CV_FOLDS, split_fold, _tune_lgbm
        train_seasons, val_season = CV_FOLDS[-1]
        train_df, val_df = split_fold(df, train_seasons, val_season)
        tuned = _tune_lgbm(
            train_df[feat_cols], train_df['total_points'],
            val_df[feat_cols],   val_df['total_points'],
            position,
        )
        log.info(f'[train] {position}/lgbm: best tuned params {tuned}')
        extra_params = tuned

    bundle = spec.build_fn(
        X, y, position,
        sid_train=s,
        tune=tune,
        extra_params=extra_params,
        _train_df=df,
    )
    bundle['model_name'] = spec.name
    bundle['position']   = position
    _save(position, spec.name, bundle, df, cv_metrics)


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
        model_names = list(get_registry().keys())
        return df[df['model'].isin(model_names)]
    return None


# ---------------------------------------------------------------------------
# Main training entry points
# ---------------------------------------------------------------------------

def train_position(
    position: str,
    models: tuple[str, ...] | None = None,
    eval_first: bool = False,
    tune: bool = False,
) -> None:
    """Train all requested models for one position."""
    registry = get_registry()
    if models is None:
        model_names = [n for n, s in registry.items() if s.family in ('tabular', 'decomposed')]
    else:
        model_names = list(models)
        for name in model_names:
            if name not in registry:
                raise ValueError(f'Unknown model "{name}". Available: {sorted(registry)}')

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

    for model_name in model_names:
        log.info(f'[train] {position}/{model_name} ...')
        spec = get_model(model_name)
        if spec.family == 'meta':
            log.warning(
                f'[train] {position}/{model_name}: meta-models cannot be trained standalone '
                f'(they require OOF base model predictions). Use ml.evaluate for CV results.'
            )
            continue
        _train_tabular(spec, df, feat_cols, position, cv_metrics, tune=tune)


def train_all(
    eval_first: bool = False,
    tune: bool = False,
) -> None:
    """Train all registered tabular models for all positions."""
    log.info('Training all positions: ' + ', '.join(VALID_POSITIONS))
    for position in VALID_POSITIONS:
        log.info(f'\n{"="*60}')
        log.info(f'Position: {position}')
        log.info('='*60)
        train_position(position, eval_first=eval_first, tune=tune)
    log.info(f'\n[done] All models serialised to {MODELS_DIR}')


# ---------------------------------------------------------------------------
# Meta-model training (simple_avg, stacking, blending)
# ---------------------------------------------------------------------------

def train_meta_position(position: str) -> None:
    """
    Build and serialise meta-model bundles for one position.

    Meta-models (simple_avg, stacking, blending) cannot be trained on a
    single full-data pass because they require out-of-fold base model
    predictions. This function reads the OOF parquet already written by
    ml.evaluate and fits each meta-learner on the full OOF stack (all 3
    folds), giving the meta-model maximum signal before serialisation.

    Requires: logs/training/cv_preds_{position}.parquet (from ml.evaluate).
    """
    from sklearn.linear_model import ElasticNet as _EN, Ridge as _Ridge
    from sklearn.preprocessing import StandardScaler as _Scaler

    oof_path = LOGS_DIR / f'cv_preds_{position}.parquet'
    if not oof_path.exists():
        log.error(
            f'[meta] {position}: OOF parquet not found at {oof_path}. '
            f'Run `python -m ml.evaluate --position {position}` first.'
        )
        return

    oof_df     = pd.read_parquet(oof_path)
    y_oof      = oof_df['total_points'].values
    cv_metrics = _load_cv_metrics(position)

    # Full feature matrix is only used for metadata (train_seasons, n_rows)
    df = build_feature_matrix(position)

    log.info(
        f'[meta] {position}: OOF rows={len(oof_df):,}, '
        f'seasons={sorted(oof_df["season_id"].unique().tolist())}'
    )

    for spec in meta_models():
        log.info(f'[meta] {position}/{spec.name} ...')

        pred_cols = [f'pred_{d}' for d in spec.deps]
        available = [d for d, c in zip(spec.deps, pred_cols) if c in oof_df.columns]
        avail_cols = [f'pred_{d}' for d in available]

        if not available:
            log.warning(
                f'[meta] {position}/{spec.name}: none of {spec.deps} found in OOF parquet; '
                f'run ml.evaluate to regenerate.'
            )
            continue

        X_meta = oof_df[avail_cols].fillna(0.0).values

        if spec.name == 'simple_avg':
            # No fitting required — equal weights across available deps
            n = len(available)
            bundle = {
                'meta_model':   None,
                'scaler':       None,
                'base_models':  available,
                'weights':      np.ones(n) / n,
                'feature_cols': available,
            }

        elif spec.name in ('stacking', 'blending'):
            scaler = _Scaler()
            X_scaled = scaler.fit_transform(X_meta)

            if spec.name == 'stacking':
                meta_model = _Ridge(alpha=0.5, random_state=42)
            else:
                meta_model = _EN(l1_ratio=0.5, alpha=0.5, max_iter=2000, random_state=42)

            meta_model.fit(X_scaled, y_oof)
            log.info(
                f'[meta] {position}/{spec.name}: fitted on {len(X_meta):,} OOF rows '
                f'({len(available)} base models)'
            )
            bundle = {
                'meta_model':   meta_model,
                'scaler':       scaler,
                'base_models':  available,
                'feature_cols': available,
            }

        else:
            log.warning(f'[meta] {position}/{spec.name}: unrecognised meta-model, skipping')
            continue

        bundle['model_name'] = spec.name
        bundle['position']   = position
        _save(position, spec.name, bundle, oof_df, cv_metrics)


def train_meta_all() -> None:
    """Build and serialise meta-model bundles for all positions."""
    log.info('Building meta-models for all positions: ' + ', '.join(VALID_POSITIONS))
    for position in VALID_POSITIONS:
        log.info(f'\n{"="*60}')
        log.info(f'Position: {position}')
        log.info('='*60)
        train_meta_position(position)
    log.info(f'\n[done] Meta-models serialised to {MODELS_DIR}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='FPL Phase 5 final model training')
    p.add_argument('--position', choices=list(VALID_POSITIONS),
                   help='Train for one position only (default: all)')
    p.add_argument('--model', choices=sorted(get_registry()),
                   help='Train one model only (default: all tabular+decomposed)')
    p.add_argument('--eval-first', action='store_true',
                   help='Run CV evaluation before training')
    p.add_argument('--tune', action='store_true',
                   help='Optuna tuning for LightGBM (uses fold 3 for search)')
    p.add_argument('--meta', action='store_true',
                   help='Train meta-models only (simple_avg, stacking, blending) '
                        'from OOF parquets in logs/training/')
    p.add_argument('--all', dest='train_all', action='store_true',
                   help='Train both tabular+decomposed models and meta-models')
    return p.parse_args()


if __name__ == '__main__':
    args   = _parse_args()
    models = (args.model,) if args.model else None

    if args.meta:
        if args.position:
            train_meta_position(args.position)
        else:
            train_meta_all()
    elif args.train_all:
        if args.position:
            train_position(args.position, models=models,
                           eval_first=args.eval_first, tune=args.tune)
            train_meta_position(args.position)
        else:
            train_all(eval_first=args.eval_first, tune=args.tune)
            train_meta_all()
    else:
        if args.position:
            train_position(
                args.position,
                models=models,
                eval_first=args.eval_first,
                tune=args.tune,
            )
        else:
            train_all(eval_first=args.eval_first, tune=args.tune)
