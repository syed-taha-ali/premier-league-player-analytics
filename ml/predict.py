"""
Phase 8: Inference pipeline — generate GW predictions from serialised models.

Loads the trained model artefacts from models/ and the current feature matrix
from outputs/features/, then produces a ranked prediction table for the
most recent available GW (or a specified GW).

All 22 registered models are available (tabular, decomposed, and meta families).
Sequential models (lstm, gru) are excluded — use ml/evaluate_sequential.py for those.
Meta-models (simple_avg, stacking, blending) automatically load and run their
base models before combining predictions.

Usage:
    python -m ml.predict                          # predict latest GW using lgbm
    python -m ml.predict --gw 25 --season 10      # specify GW and season
    python -m ml.predict --model ridge            # single model
    python -m ml.predict --models ridge xgb lgbm  # ensemble of named models
    python -m ml.predict --top 20                 # show top-20 per position

Outputs:
    outputs/predictions/gw{N}_s{season}_predictions.csv
    Console: top-5 per position
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.features import build_feature_matrix, CONTEXT_COLS, TARGET_COL, VALID_POSITIONS
from ml.evaluate import get_feature_cols
from ml.models import get_registry, get_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

_HERE        = Path(__file__).parent.parent
MODELS_DIR   = _HERE / 'models'
OUTPUTS_PRED = _HERE / 'outputs' / 'predictions'



# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(position: str, model_name: str) -> dict:
    """Load a serialised model bundle from models/{position}_{model_name}.pkl."""
    path = MODELS_DIR / f'{position}_{model_name}.pkl'
    if not path.exists():
        raise FileNotFoundError(
            f'No trained model found at {path}. '
            f'Run `python -m ml.train --position {position} --model {model_name}` first.'
        )
    bundle = joblib.load(path)
    log.info(f'[load] {path}')
    return bundle


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_position(
    position: str,
    model_name: str,
    df_predict: pd.DataFrame,
) -> pd.Series:
    """
    Generate predictions for a position using the specified model.

    Parameters
    ----------
    position    : one of VALID_POSITIONS
    model_name  : registered model name (see ml.models.get_registry())
    df_predict  : feature matrix rows to predict (from build_feature_matrix)

    Returns
    -------
    pd.Series of predicted points, aligned to df_predict index.
    """
    spec       = get_model(model_name)
    bundle     = load_model(position, model_name)
    feat_cols  = get_feature_cols(df_predict)
    X          = df_predict[feat_cols]
    season_ids = df_predict['season_id']

    if spec.family == 'meta':
        # Run each base model first, then feed predictions into the meta predict_fn
        dep_preds: dict[str, np.ndarray] = {}
        for dep in spec.deps:
            try:
                dep_bundle = load_model(position, dep)
                dep_spec   = get_model(dep)
                dep_preds[dep] = dep_spec.predict_fn(
                    dep_bundle, X, sid=season_ids, _df=df_predict
                )
            except FileNotFoundError:
                log.warning(f'[predict] Base model {dep} not found for meta-model {model_name}')
                dep_preds[dep] = np.full(len(X), np.nan)
        preds = spec.predict_fn(
            bundle, X, sid=season_ids, _dep_preds=dep_preds, _df=df_predict
        )
    else:
        preds = spec.predict_fn(bundle, X, sid=season_ids, _df=df_predict)

    return pd.Series(preds, index=df_predict.index, name=f'pred_{model_name}')


def _predict_bayesian_ridge_std(
    position: str,
    df_predict: pd.DataFrame,
) -> pd.Series:
    """
    Return per-row prediction std from the BayesianRidge model.
    Replicates the imputation/scaling steps of _predict_scaled_linear, then
    calls model.predict(return_std=True) to get the posterior std.
    """
    bundle       = load_model(position, 'bayesian_ridge')
    feat_cols    = bundle['feature_cols']
    season_means = bundle['season_means']
    global_means = bundle['global_means']
    scaler       = bundle['scaler']
    model        = bundle['model']
    sid          = df_predict['season_id']

    X_aligned = df_predict[feat_cols].reset_index(drop=True)
    X_arr     = X_aligned.values.astype(float)
    gm        = global_means[feat_cols].values

    s_vals = sid.reset_index(drop=True).values
    fill   = np.empty_like(X_arr)
    for i, s in enumerate(s_vals):
        if s in season_means.index:
            fill[i] = season_means.loc[s, feat_cols].values
        else:
            fill[i] = gm
    fill     = np.where(np.isnan(fill), gm, fill)
    nan_mask = np.isnan(X_arr)
    X_filled = np.where(nan_mask, fill, X_arr)
    X_filled = np.where(np.isnan(X_filled), gm, X_filled)

    _, std = model.predict(scaler.transform(X_filled), return_std=True)
    return pd.Series(std, index=df_predict.index, name='pred_bayesian_ridge_std')


# ---------------------------------------------------------------------------
# GW prediction entry point
# ---------------------------------------------------------------------------

def predict_gw(
    gw: int | None = None,
    season_id: int | None = None,
    models: tuple[str, ...] = ('lgbm',),
    top_n_print: int = 5,
) -> pd.DataFrame:
    """
    Predict for a specific GW (or the latest available GW in the feature matrix).

    Returns a DataFrame with one row per player, columns:
        player_code, position, season_id, gw, team_sk,
        total_points (actual, if available),
        pred_{model} for each model requested,
        pred_ensemble (mean across models, if >1 model).
    """
    all_rows = []

    for position in VALID_POSITIONS:
        df = build_feature_matrix(position)

        # Identify target GW
        if season_id is not None and gw is not None:
            df_gw = df[(df['season_id'] == season_id) & (df['gw'] == gw)].copy()
        elif season_id is not None:
            latest_gw = df[df['season_id'] == season_id]['gw'].max()
            df_gw = df[(df['season_id'] == season_id) & (df['gw'] == latest_gw)].copy()
        else:
            # Latest GW across all seasons
            latest_sid = df['season_id'].max()
            latest_gw  = df[df['season_id'] == latest_sid]['gw'].max()
            df_gw = df[(df['season_id'] == latest_sid) & (df['gw'] == latest_gw)].copy()

        if df_gw.empty:
            log.warning(f'[predict] {position}: no rows for season={season_id} gw={gw}')
            continue

        df_gw = df_gw.reset_index(drop=True)
        df_gw['position'] = position

        for model_name in models:
            try:
                df_gw[f'pred_{model_name}'] = predict_position(position, model_name, df_gw)
                if model_name == 'bayesian_ridge':
                    df_gw['pred_bayesian_ridge_std'] = _predict_bayesian_ridge_std(
                        position, df_gw
                    )
            except FileNotFoundError as e:
                log.error(str(e))
                df_gw[f'pred_{model_name}'] = np.nan

        all_rows.append(df_gw)

    if not all_rows:
        log.error('No predictions generated.')
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # Ensemble: mean over available model predictions
    pred_cols = [f'pred_{m}' for m in models if f'pred_{m}' in combined.columns]
    if len(pred_cols) > 1:
        combined['pred_ensemble'] = combined[pred_cols].mean(axis=1)
        rank_col = 'pred_ensemble'
    else:
        rank_col = pred_cols[0] if pred_cols else None

    # Sort by best prediction
    if rank_col:
        combined = combined.sort_values(rank_col, ascending=False)

    # Save to CSV
    _save_predictions(combined, gw, season_id)

    # Print top-N per position
    if rank_col:
        _print_top_n(combined, rank_col, top_n_print)

    return combined


def _save_predictions(df: pd.DataFrame, gw: int | None, season_id: int | None) -> None:
    OUTPUTS_PRED.mkdir(parents=True, exist_ok=True)
    gw_tag      = gw or df['gw'].max()
    season_tag  = season_id or df['season_id'].max()
    fname       = f'gw{gw_tag}_s{season_tag}_predictions.csv'
    out         = OUTPUTS_PRED / fname
    df.to_csv(out, index=False)
    log.info(f'[save] Predictions -> {out}')


def _print_top_n(df: pd.DataFrame, rank_col: str, n: int) -> None:
    print(f'\n{"="*60}')
    print(f'Top {n} predictions per position (ranked by {rank_col})')
    print('='*60)
    display_cols = ['player_code', 'position', 'gw', 'season_id', rank_col]
    if 'total_points' in df.columns:
        display_cols.append('total_points')

    for position in VALID_POSITIONS:
        subset = df[df['position'] == position].head(n)
        if subset.empty:
            continue
        print(f'\n{position}:')
        print(subset[[c for c in display_cols if c in subset.columns]].to_string(index=False))


# ---------------------------------------------------------------------------
# Holdout evaluation: run predictions on known past GWs and compare to actuals
# ---------------------------------------------------------------------------

def evaluate_predictions(
    season_id: int,
    model_name: str = 'lgbm',
) -> pd.DataFrame:
    """
    For each GW in the given season, predict using that GW's features and
    compare to actual total_points. Returns per-GW MAE and Spearman rho.

    Useful for quick sanity-check of trained models on historical data.
    """
    from scipy import stats
    from sklearn.metrics import mean_absolute_error

    records = []
    for position in VALID_POSITIONS:
        df = build_feature_matrix(position)
        df_season = df[df['season_id'] == season_id].copy()
        if df_season.empty:
            continue

        # Load model once per position, not once per GW
        try:
            bundle = load_model(position, model_name)
        except FileNotFoundError as e:
            log.error(str(e))
            continue

        spec = get_model(model_name)
        for gw in sorted(df_season['gw'].unique()):
            df_gw = df_season[df_season['gw'] == gw].copy().reset_index(drop=True)
            try:
                feat_cols  = get_feature_cols(df_gw)
                X          = df_gw[feat_cols]
                preds      = spec.predict_fn(bundle, X, sid=df_gw['season_id'])
            except Exception:
                continue
            actuals = df_gw['total_points'].values
            mae     = mean_absolute_error(actuals, preds)
            rho, _  = stats.spearmanr(actuals, preds)
            records.append({
                'position': position,
                'gw':       gw,
                'n':        len(df_gw),
                'mae':      round(mae, 4),
                'spearman': round(rho, 4),
            })

    result = pd.DataFrame(records)
    if not result.empty:
        print(f'\nPer-GW evaluation (season {season_id}, model={model_name})')
        summary = result.groupby('position')[['mae', 'spearman']].mean().round(4)
        print(summary.to_string())
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='FPL GW prediction inference')
    p.add_argument('--gw',     type=int, help='Gameweek number to predict')
    p.add_argument('--season', type=int, help='Season ID (e.g. 10 = 2025-26)')
    p.add_argument('--model',  choices=sorted(get_registry()),
                   help='Model to use (default: lgbm). Pass multiple via repeated flag.')
    p.add_argument('--models', nargs='+', choices=sorted(get_registry()),
                   help='One or more models (overrides --model)')
    p.add_argument('--top',    type=int, default=5, help='Top-N to print per position')
    p.add_argument('--eval',   type=int, metavar='SEASON_ID',
                   help='Run per-GW holdout evaluation for a season')
    return p.parse_args()


if __name__ == '__main__':
    args   = _parse_args()
    models = tuple(args.models) if args.models else (args.model or 'lgbm',)

    if args.eval is not None:
        evaluate_predictions(args.eval, model_name=models[0])
    else:
        predict_gw(
            gw=args.gw,
            season_id=args.season,
            models=models,
            top_n_print=args.top,
        )
