"""
Phase 8: End-to-end GW runner.

Orchestrates the post-GW operational workflow:
  Step 1 — Fetch    : pull latest data from FPL API, update CSVs
  Step 2 — ETL      : rebuild db/fpl.db from scratch
  Step 3 — Predict  : generate predictions for the GW
  Step 4 — Monitor  : compare predictions vs actuals, log metrics, alert if above threshold

Usage:
    python run_gw.py                                          # fetch current GW, all steps
    python run_gw.py --gw 25 --season 10                     # explicit GW and season
    python run_gw.py --gw 25 --season 10 --skip-fetch        # skip API call, use existing CSVs
    python run_gw.py --gw 25 --season 10 --skip-etl          # skip ETL rebuild
    python run_gw.py --gw 25 --season 10 --model ridge lgbm  # override models
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

_HERE       = Path(__file__).parent
_MONITOR_DIR = _HERE / 'logs' / 'monitoring'
_MONITOR_CSV = _MONITOR_DIR / 'monitoring_log.csv'

# Alert thresholds: 1.5 × baseline MAE (from Phase 6 evaluation)
_THRESHOLDS = {
    'GK':  3.494,
    'DEF': 3.498,
    'MID': 2.996,
    'FWD': 3.609,
}

# Default models to run
_DEFAULT_MODELS = ('ridge', 'bayesian_ridge', 'blending')

# Season label lookup (season_id -> label used in data/ directory)
_SEASON_LABELS = {
    1: '2016-17', 2: '2017-18', 3: '2018-19', 4: '2019-20',
    5: '2020-21', 6: '2021-22', 7: '2022-23', 8: '2023-24',
    9: '2024-25', 10: '2025-26',
}


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _step_fetch(gw: int, season_id: int) -> None:
    """Fetch FPL API data and write updated CSVs."""
    from etl.fetch import (
        fetch_bootstrap, fetch_fixtures, fetch_gw_live,
        write_season_csvs, DATA_DIR,
    )
    season_label = _SEASON_LABELS.get(season_id, '2025-26')
    log.info(f'[step1] Fetching GW {gw} data from FPL API ...')
    bootstrap = fetch_bootstrap()
    fixtures  = fetch_fixtures()
    live      = fetch_gw_live(gw)
    write_season_csvs(season_label, gw, bootstrap, fixtures, live, DATA_DIR)
    log.info(f'[step1] Done — CSVs updated for {season_label} GW {gw}')


def _step_etl() -> None:
    """Rebuild fpl.db from scratch via etl/run.py."""
    log.info('[step2] Rebuilding fpl.db ...')
    result = subprocess.run(
        [sys.executable, '-m', 'etl.run'],
        cwd=str(_HERE),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError('[step2] ETL failed — aborting GW run.')
    log.info('[step2] ETL complete — all 11 validation checks passed')


def _step_predict(gw: int, season_id: int, models: tuple[str, ...]) -> pd.DataFrame:
    """Generate predictions for the GW using ml.predict."""
    log.info(f'[step3] Generating predictions for GW {gw} season {season_id} ...')
    from ml.predict import predict_gw
    df = predict_gw(gw=gw, season_id=season_id, models=models)
    if df.empty:
        log.warning('[step3] predict_gw returned an empty DataFrame.')
    else:
        log.info(f'[step3] Predictions generated: {len(df):,} rows')
    return df


def _step_monitor(
    gw: int,
    season_id: int,
    predictions: pd.DataFrame,
    primary_model: str = 'ridge',
) -> None:
    """
    Compare predictions vs actuals, compute metrics, append to monitoring log.

    Only runs if actual total_points are available (GW is complete).
    """
    log.info('[step4] Running monitoring checks ...')

    if predictions.empty:
        log.warning('[step4] No predictions to evaluate.')
        return

    pred_col = f'pred_{primary_model}'
    if pred_col not in predictions.columns:
        log.warning(f'[step4] Column {pred_col} not in predictions — skipping monitoring.')
        return

    if 'total_points' not in predictions.columns:
        log.warning('[step4] total_points not in predictions — GW may not be complete yet.')
        return

    actuals = predictions['total_points']
    if actuals.isna().all():
        log.warning('[step4] All total_points are NaN — GW not yet complete.')
        return

    _MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict] = []

    for position in ('GK', 'DEF', 'MID', 'FWD'):
        mask = predictions['position'] == position
        sub  = predictions[mask].dropna(subset=[pred_col, 'total_points'])
        if len(sub) < 2:
            continue

        y_true = sub['total_points'].values
        y_pred = sub[pred_col].values

        mae        = float(mean_absolute_error(y_true, y_pred))
        rmse       = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        rho, _     = stats.spearmanr(y_true, y_pred)
        spearman   = float(rho) if not np.isnan(rho) else 0.0

        # Top-10 precision: fraction of true top-10 in predicted top-10
        n_top = min(10, len(sub))
        true_top  = set(sub.nlargest(n_top, 'total_points').index)
        pred_top  = set(sub.nlargest(n_top, pred_col).index)
        top10_prec = len(true_top & pred_top) / n_top

        new_rows.append({
            'season_id':     season_id,
            'gw':            gw,
            'model':         primary_model,
            'position':      position,
            'mae':           round(mae, 4),
            'rmse':          round(rmse, 4),
            'spearman':      round(spearman, 4),
            'top10_precision': round(top10_prec, 4),
            'rolling_mae_5gw': None,  # filled below
            'threshold':     _THRESHOLDS.get(position),
            'alert':         0,
            'logged_at':     datetime.now(timezone.utc).isoformat(),
        })

    if not new_rows:
        log.warning('[step4] No per-position metrics computed.')
        return

    # Load existing log and compute 5-GW rolling MAE
    if _MONITOR_CSV.exists():
        existing = pd.read_csv(_MONITOR_CSV)
    else:
        existing = pd.DataFrame(columns=list(new_rows[0].keys()))

    new_df = pd.DataFrame(new_rows)

    for row in new_rows:
        pos   = row['position']
        model = row['model']
        hist  = existing[
            (existing['position'] == pos) & (existing['model'] == model)
        ]['mae'].tail(4).tolist()  # last 4 + current = 5
        hist.append(row['mae'])
        rolling = round(float(np.mean(hist)), 4)
        row['rolling_mae_5gw'] = rolling

        threshold = _THRESHOLDS.get(pos, float('inf'))
        if rolling > threshold:
            row['alert'] = 1
            log.warning(
                f'[monitor] ALERT: {pos}/{model} rolling 5-GW MAE {rolling:.4f} '
                f'exceeds threshold {threshold:.4f}'
            )

    new_df = pd.DataFrame(new_rows)
    updated = pd.concat([existing, new_df], ignore_index=True)
    updated.to_csv(_MONITOR_CSV, index=False)
    log.info(f'[step4] Monitoring log updated: {_MONITOR_CSV}')

    # Summary
    print('\n--- Monitoring summary ---')
    for row in new_rows:
        alert_flag = ' [ALERT]' if row['alert'] else ''
        print(
            f"  {row['position']:3s}  MAE={row['mae']:.4f}  "
            f"rolling={row['rolling_mae_5gw']:.4f}  "
            f"threshold={row['threshold']:.4f}{alert_flag}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    gw: int | None = None,
    season_id: int = 10,
    models: tuple[str, ...] = _DEFAULT_MODELS,
    skip_fetch: bool = False,
    skip_etl: bool = False,
    primary_model: str = 'ridge',
) -> None:
    """Execute the full post-GW pipeline."""
    # Resolve GW number if not specified
    if gw is None:
        if not skip_fetch:
            from etl.fetch import fetch_bootstrap, get_current_gw
            bootstrap = fetch_bootstrap()
            gw = get_current_gw(bootstrap)
            log.info(f'[run] Auto-detected current GW: {gw}')
        else:
            raise ValueError('--gw is required when --skip-fetch is set.')

    log.info(f'[run] GW={gw}  season={season_id}  models={models}')

    if not skip_fetch:
        _step_fetch(gw, season_id)
    else:
        log.info('[step1] Skipped (--skip-fetch)')

    if not skip_etl:
        _step_etl()
    else:
        log.info('[step2] Skipped (--skip-etl)')

    predictions = _step_predict(gw, season_id, models)
    _step_monitor(gw, season_id, predictions, primary_model=primary_model)

    log.info(f'[run] GW {gw} pipeline complete.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='FPL end-to-end GW runner')
    p.add_argument('--gw',     type=int,
                   help='GW number to process (default: current finished GW from API)')
    p.add_argument('--season', type=int, default=10,
                   help='Season ID (default: 10 = 2025-26)')
    p.add_argument('--model',  nargs='+', dest='models',
                   default=list(_DEFAULT_MODELS),
                   help='Models to run predictions for (default: ridge bayesian_ridge blending)')
    p.add_argument('--skip-fetch', action='store_true',
                   help='Skip API fetch; use existing CSV data')
    p.add_argument('--skip-etl', action='store_true',
                   help='Skip ETL rebuild; use existing fpl.db')
    p.add_argument('--primary-model', default='ridge',
                   help='Model used for monitoring metrics (default: ridge)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(
        gw=args.gw,
        season_id=args.season,
        models=tuple(args.models),
        skip_fetch=args.skip_fetch,
        skip_etl=args.skip_etl,
        primary_model=args.primary_model,
    )
