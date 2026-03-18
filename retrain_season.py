"""
End-of-season retraining pipeline.

Orchestrates all steps required to incorporate a new FPL season into the model
registry: archiving the previous season's artefacts, rebuilding the DB, clearing
stale caches, running CV evaluation, alpha search, full model training, and
meta-model training.

Usage:
    python retrain_season.py --season 11
    python retrain_season.py --season 11 --dry-run
    python retrain_season.py --season 11 --skip-archive --skip-etl

Prerequisites (manual steps before running):
    1. Add new season row to SEASONS in etl/schema.py (season metadata, era flags)
    2. Add season label to _SEASON_LABELS in run_gw.py
    3. Add season to SEASON_NAME_TO_ID in etl/schema.py
    4. Confirm data/{new_season}/gws/merged_gw.csv exists and is complete

Steps:
    1  Verify   Check data directory and CSV prerequisites
    2  Archive  Copy models/*.pkl and models/*_meta.json to models/v{prev_season}/
    3  ETL      Rebuild db/fpl.db from scratch
    4  Cache    Delete outputs/features/*.parquet (stale after new season added)
    5  Evaluate Run expanding-window CV on new fold set; write cv_metrics, OOF parquets
    6  Alpha    Run Ridge alpha grid search to find best alpha per position
    7  Train    Retrain all tabular + decomposed models
    8  Meta     Rebuild meta-models (simple_avg, stacking, blending) from OOF predictions
    9  Report   Write logs/training/retrain_s{season}_report.md with new CV MAE per position
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

_HERE        = Path(__file__).parent
_MODELS_DIR  = _HERE / 'models'
_OUTPUTS_DIR = _HERE / 'outputs' / 'features'
_LOGS_DIR    = _HERE / 'logs' / 'training'
_DATA_DIR    = _HERE / 'data'


# ---------------------------------------------------------------------------
# Prerequisite validation
# ---------------------------------------------------------------------------

def _validate_prerequisites(season: int) -> tuple[str, int]:
    """
    Validate that the new season is ready to train on.

    Returns
    -------
    (season_label, prev_season_id) where season_label is the data directory
    name (e.g. '2026-27') and prev_season_id is season - 1.

    Raises RuntimeError if any prerequisite fails.
    """
    from etl.schema import SEASONS

    season_row = next((s for s in SEASONS if s[0] == season), None)
    if season_row is None:
        raise RuntimeError(
            f'Season {season} not found in etl/schema.py SEASONS. '
            f'Add the season row before running retrain_season.py.'
        )

    season_label = season_row[1]
    csv_path     = _DATA_DIR / season_label / 'gws' / 'merged_gw.csv'
    pr_path      = _DATA_DIR / season_label / 'players_raw.csv'

    if not csv_path.exists():
        raise RuntimeError(
            f'merged_gw.csv not found at {csv_path}. '
            f'Ensure the season data is complete before retraining.'
        )
    if not pr_path.exists():
        raise RuntimeError(
            f'players_raw.csv not found at {pr_path}. '
            f'Ensure the season data is complete before retraining.'
        )

    prev_season = season - 1
    log.info(f'[verify] Season {season} ({season_label}) prerequisites satisfied')
    return season_label, prev_season


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _step_archive(prev_season: int, dry_run: bool) -> None:
    """Copy current model artefacts to models/v{prev_season}/."""
    archive_dir = _MODELS_DIR / f'v{prev_season}'
    log.info(f'[archive] Archiving models to {archive_dir}')
    if dry_run:
        log.info(f'[dry-run] Would create {archive_dir} and copy *.pkl, *_meta.json')
        return

    archive_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in ('*.pkl', '*_meta.json'):
        for src in _MODELS_DIR.glob(pattern):
            shutil.copy2(src, archive_dir / src.name)
            copied += 1

    log.info(f'[archive] Archived {copied} artefacts to {archive_dir}')


def _step_etl(dry_run: bool) -> None:
    """Rebuild fpl.db from scratch."""
    log.info('[etl] Rebuilding fpl.db ...')
    if dry_run:
        log.info('[dry-run] Would run: python -m etl.run')
        return

    result = subprocess.run(
        [sys.executable, '-m', 'etl.run'],
        cwd=str(_HERE),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError('[etl] ETL failed -- aborting retrain.')
    log.info('[etl] ETL complete')


def _step_clear_cache(dry_run: bool) -> None:
    """Delete stale feature matrix parquet files."""
    parquets = list(_OUTPUTS_DIR.glob('*.parquet'))
    log.info(f'[cache] Deleting {len(parquets)} stale parquet files from {_OUTPUTS_DIR}')
    if dry_run:
        for p in parquets:
            log.info(f'[dry-run] Would delete {p.name}')
        return

    for p in parquets:
        p.unlink()
    log.info('[cache] Feature matrix cache cleared')


def _step_evaluate(dry_run: bool) -> None:
    """Run full CV evaluation on new fold set."""
    log.info('[evaluate] Running CV evaluation (all positions) ...')
    if dry_run:
        log.info('[dry-run] Would run: python -m ml.evaluate')
        return

    result = subprocess.run(
        [sys.executable, '-m', 'ml.evaluate'],
        cwd=str(_HERE),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError('[evaluate] CV evaluation failed -- aborting retrain.')
    log.info('[evaluate] CV evaluation complete')


def _step_alpha_search(dry_run: bool) -> None:
    """Run Ridge alpha grid search to find best alpha per position."""
    log.info('[alpha] Running Ridge alpha grid search ...')
    if dry_run:
        log.info('[dry-run] Would run: python -m ml.train --model ridge --alpha-search')
        return

    result = subprocess.run(
        [sys.executable, '-m', 'ml.train', '--model', 'ridge', '--alpha-search'],
        cwd=str(_HERE),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError('[alpha] Alpha search failed -- aborting retrain.')
    log.info('[alpha] Alpha search complete')


def _step_train_all(dry_run: bool) -> None:
    """Retrain all tabular and decomposed models."""
    log.info('[train] Retraining all models ...')
    if dry_run:
        log.info('[dry-run] Would run: python -m ml.train --all')
        return

    result = subprocess.run(
        [sys.executable, '-m', 'ml.train', '--all'],
        cwd=str(_HERE),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError('[train] Model training failed -- aborting retrain.')
    log.info('[train] All models trained')


def _step_train_meta(dry_run: bool) -> None:
    """Rebuild meta-models from OOF predictions."""
    log.info('[meta] Rebuilding meta-models ...')
    if dry_run:
        log.info('[dry-run] Would run: python -m ml.train --meta')
        return

    result = subprocess.run(
        [sys.executable, '-m', 'ml.train', '--meta'],
        cwd=str(_HERE),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError('[meta] Meta-model training failed -- aborting retrain.')
    log.info('[meta] Meta-models rebuilt')


def _step_report(season: int, dry_run: bool) -> None:
    """Write retraining report with new CV MAE per position."""
    report_path = _LOGS_DIR / f'retrain_s{season}_report.md'
    log.info(f'[report] Writing retrain report to {report_path}')
    if dry_run:
        log.info(f'[dry-run] Would write {report_path}')
        return

    import pandas as pd

    lines = [
        f'# Retraining Report -- Season {season}',
        '',
        f'**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}',
        '',
        '## CV Results (mean across all folds)',
        '',
    ]

    positions = ('GK', 'DEF', 'MID', 'FWD')
    summary_rows = []
    for pos in positions:
        csv_path = _LOGS_DIR / f'cv_metrics_{pos}.csv'
        if not csv_path.exists():
            summary_rows.append({'Position': pos, 'MAE': 'N/A', 'RMSE': 'N/A', 'Spearman': 'N/A'})
            continue
        df = pd.read_csv(csv_path)
        ridge_rows = df[df['model'] == 'ridge']
        if ridge_rows.empty:
            summary_rows.append({'Position': pos, 'MAE': 'N/A', 'RMSE': 'N/A', 'Spearman': 'N/A'})
            continue
        summary_rows.append({
            'Position': pos,
            'MAE':      round(ridge_rows['mae'].mean(), 4),
            'RMSE':     round(ridge_rows['rmse'].mean(), 4),
            'Spearman': round(ridge_rows['spearman'].mean(), 4),
        })

    lines += [
        '| Position | Ridge MAE | Ridge RMSE | Ridge Spearman |',
        '|----------|-----------|------------|----------------|',
    ]
    for r in summary_rows:
        lines.append(f"| {r['Position']} | {r['MAE']} | {r['RMSE']} | {r['Spearman']} |")

    lines += [
        '',
        '## Steps Completed',
        '',
        '1. Prerequisites verified',
        f'2. Previous season models archived to models/v{season - 1}/',
        '3. ETL rebuilt (fpl.db)',
        '4. Feature matrix cache cleared',
        '5. CV evaluation run',
        '6. Ridge alpha grid search run',
        '7. All models retrained',
        '8. Meta-models rebuilt',
        '9. This report written',
        '',
        '## Next Steps',
        '',
        '- Review CV MAE changes vs previous season',
        '- Update monitoring alert thresholds if baseline MAE has shifted significantly',
        '- Update CLAUDE.md Phase 8 key decisions section with new CV results',
        '- Run run_gw.py for the first GW of the new season to confirm pipeline is operational',
    ]

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n')
    log.info(f'[report] Report written to {report_path}')


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def retrain(season: int, skip_archive: bool, skip_etl: bool, dry_run: bool) -> None:
    """Run the full end-of-season retraining pipeline."""
    log.info(f'[retrain] Starting end-of-season retraining for season {season}')
    if dry_run:
        log.info('[retrain] DRY RUN -- no changes will be made')

    # Step 1: Validate prerequisites (in dry-run, show what would be checked)
    if dry_run:
        log.info(
            f'[dry-run] Would validate: etl/schema.py contains season {season}, '
            f'data/{{label}}/gws/merged_gw.csv exists, '
            f'data/{{label}}/players_raw.csv exists'
        )
        season_label = f'(season {season} label)'
        prev_season  = season - 1
    else:
        season_label, prev_season = _validate_prerequisites(season)
    log.info(f'[retrain] New season: {season} ({season_label})  Previous: {prev_season}')

    # Step 2: Archive
    if not skip_archive:
        _step_archive(prev_season, dry_run)
    else:
        log.info('[archive] Skipped (--skip-archive)')

    # Step 3: ETL
    if not skip_etl:
        _step_etl(dry_run)
    else:
        log.info('[etl] Skipped (--skip-etl)')

    # Step 4: Clear feature cache
    _step_clear_cache(dry_run)

    # Step 5: CV evaluation
    _step_evaluate(dry_run)

    # Step 6: Ridge alpha search
    _step_alpha_search(dry_run)

    # Step 7: Train all models
    _step_train_all(dry_run)

    # Step 8: Rebuild meta-models
    _step_train_meta(dry_run)

    # Step 9: Write report
    _step_report(season, dry_run)

    log.info(f'[retrain] Season {season} retraining complete.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='FPL end-of-season retraining pipeline')
    p.add_argument('--season', type=int, required=True,
                   help='New season ID being introduced (e.g. 11)')
    p.add_argument('--skip-archive', action='store_true',
                   help='Skip model archiving (if already done)')
    p.add_argument('--skip-etl', action='store_true',
                   help='Skip ETL rebuild')
    p.add_argument('--dry-run', action='store_true',
                   help='Print steps without executing any changes')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    retrain(
        season=args.season,
        skip_archive=args.skip_archive,
        skip_etl=args.skip_etl,
        dry_run=args.dry_run,
    )
