"""
Phase 8: FPL API client and CSV writer.

Fetches live data from the FPL public API and writes/updates the CSV files
that etl/run.py consumes. No database writes happen here — fetch.py only
produces CSV artefacts in data/{season}/.

Usage (standalone):
    python -m etl.fetch                           # fetch current GW
    python -m etl.fetch --gw 25                   # fetch specific GW
    python -m etl.fetch --gw 25 --season 2025-26  # explicit season label

Public API:
    fetch_bootstrap()           -> dict
    fetch_fixtures()            -> list[dict]
    fetch_gw_live(gw)           -> dict
    fetch_player_summary(eid)   -> dict
    build_merged_gw(...)        -> pd.DataFrame
    write_season_csvs(...)      -> None
    get_current_gw(bootstrap)   -> int
    get_next_gw(bootstrap)      -> int
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

_BASE_URL = 'https://fantasy.premierleague.com/api'
_TIMEOUT  = 30   # seconds per request
_RETRIES  = 3    # attempts before raising FetchError
_BACKOFF  = 2.0  # base seconds for exponential backoff

_HERE     = Path(__file__).parent.parent
DATA_DIR  = _HERE / 'data'

# FPL API element_type -> position label
_ELEMENT_TYPE_MAP = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD', 5: 'MID'}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FetchError(RuntimeError):
    """Raised when a critical FPL API endpoint returns a non-200 response."""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, critical: bool = True) -> Any:
    """
    GET url with exponential-backoff retry.

    Parameters
    ----------
    url      : full URL to fetch
    critical : if True, raise FetchError on final failure; otherwise return None
    """
    last_exc: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.get(url, timeout=_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            msg = f'HTTP {resp.status_code} from {url}'
            if critical:
                if attempt == _RETRIES:
                    raise FetchError(msg)
                log.warning(f'[fetch] {msg} — retry {attempt}/{_RETRIES}')
            else:
                log.warning(f'[fetch] {msg} (non-critical, attempt {attempt})')
                if attempt == _RETRIES:
                    return None
        except requests.RequestException as exc:
            last_exc = exc
            log.warning(f'[fetch] Request error on attempt {attempt}/{_RETRIES}: {exc}')
            if attempt == _RETRIES:
                if critical:
                    raise FetchError(f'Failed after {_RETRIES} attempts: {url}') from exc
                return None
        wait = _BACKOFF ** attempt
        log.info(f'[fetch] Waiting {wait:.1f}s before retry ...')
        time.sleep(wait)

    if last_exc and critical:
        raise FetchError(f'Failed after {_RETRIES} attempts: {url}') from last_exc
    return None


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_bootstrap() -> dict:
    """GET /bootstrap-static/ — players, teams, events."""
    log.info('[fetch] bootstrap-static ...')
    return _get(f'{_BASE_URL}/bootstrap-static/', critical=True)


def fetch_fixtures() -> list[dict]:
    """GET /fixtures/ — all fixtures across the season."""
    log.info('[fetch] fixtures ...')
    return _get(f'{_BASE_URL}/fixtures/', critical=True)


def fetch_gw_live(gw: int) -> dict:
    """GET /event/{gw}/live/ — player stats for a single GW."""
    log.info(f'[fetch] event/{gw}/live ...')
    return _get(f'{_BASE_URL}/event/{gw}/live/', critical=True)


def fetch_player_summary(element_id: int) -> dict | None:
    """GET /element-summary/{element_id}/ — player career history (non-critical)."""
    return _get(f'{_BASE_URL}/element-summary/{element_id}/', critical=False)


def get_current_gw(bootstrap: dict) -> int:
    """Return the most recently finished GW number."""
    events = bootstrap['events']
    finished = [e for e in events if e.get('finished')]
    if not finished:
        raise FetchError('No finished GWs found in bootstrap events.')
    return max(e['id'] for e in finished)


def get_next_gw(bootstrap: dict) -> int:
    """Return the next unstarted GW number."""
    events = bootstrap['events']
    unstarted = [e for e in events if not e.get('finished') and not e.get('is_current')]
    if not unstarted:
        raise FetchError('No upcoming GWs found in bootstrap events.')
    return min(e['id'] for e in unstarted)


# ---------------------------------------------------------------------------
# Data transformation
# ---------------------------------------------------------------------------

def build_merged_gw(
    gw: int,
    bootstrap: dict,
    fixtures: list[dict],
    live: dict,
) -> pd.DataFrame:
    """
    Transform FPL API responses into a DataFrame matching the merged_gw.csv schema.

    One row per player per fixture. Double GWs produce two rows per player.
    Manager rows (element_type=5 in some seasons) are included with mng_* columns.

    Parameters
    ----------
    gw        : gameweek number
    bootstrap : response from /bootstrap-static/
    fixtures  : response from /fixtures/
    live      : response from /event/{gw}/live/

    Returns
    -------
    pd.DataFrame matching merged_gw.csv column schema for the xG/defensive era.
    """
    # --- Index reference data ---
    elements_by_id: dict[int, dict] = {e['id']: e for e in bootstrap['elements']}
    teams_by_id:    dict[int, dict] = {t['id']: t for t in bootstrap['teams']}

    # Fixtures for this GW only (may be multiple in a DGW)
    gw_fixtures = [f for f in fixtures if f.get('event') == gw and f.get('finished')]
    if not gw_fixtures:
        # Include started but not yet finished (live GW)
        gw_fixtures = [f for f in fixtures if f.get('event') == gw]

    # Map: element_id -> list of fixture rows the player appeared in
    # We'll join player stats with fixture context using team membership.
    fixture_by_team: dict[int, list[dict]] = {}
    for fx in gw_fixtures:
        fixture_by_team.setdefault(fx['team_h'], []).append(fx)
        fixture_by_team.setdefault(fx['team_a'], []).append(fx)

    rows: list[dict] = []
    for el_entry in live.get('elements', []):
        element_id = el_entry['id']
        stats      = el_entry.get('stats', {})

        if element_id not in elements_by_id:
            log.warning(f'[build] element {element_id} not in bootstrap — skipping')
            continue

        el = elements_by_id[element_id]
        team_id    = el.get('team')
        team_info  = teams_by_id.get(team_id, {})
        team_name  = team_info.get('name', '')
        position   = _ELEMENT_TYPE_MAP.get(el.get('element_type', 0), '')

        player_fixtures = fixture_by_team.get(team_id, [])
        if not player_fixtures:
            # Player's team has no fixture this GW (blank GW) — skip
            continue

        for fx in player_fixtures:
            is_home  = fx['team_h'] == team_id
            opp_id   = fx['team_a'] if is_home else fx['team_h']
            opp_name = teams_by_id.get(opp_id, {}).get('name', '')

            # In a DGW the stats from /live/ aggregate across all fixtures.
            # For single GWs this is straightforward.
            row: dict = {
                'name':            el.get('web_name', ''),
                'position':        position,
                'team':            team_name,
                'xP':              stats.get('expected_points', None),
                'assists':         stats.get('assists', 0),
                'bonus':           stats.get('bonus', 0),
                'bps':             stats.get('bps', 0),
                'clean_sheets':    stats.get('clean_sheets', 0),
                'clearances_blocks_interceptions': stats.get(
                    'clearances_blocks_interceptions', None),
                'creativity':      stats.get('creativity', 0.0),
                'defensive_contribution': stats.get('defensive_contribution', None),
                'element':         element_id,
                'expected_assists':             stats.get('expected_assists', None),
                'expected_goal_involvements':   stats.get('expected_goal_involvements', None),
                'expected_goals':               stats.get('expected_goals', None),
                'expected_goals_conceded':      stats.get('expected_goals_conceded', None),
                'fixture':         fx['id'],
                'goals_conceded':  stats.get('goals_conceded', 0),
                'goals_scored':    stats.get('goals_scored', 0),
                'ict_index':       stats.get('ict_index', 0.0),
                'influence':       stats.get('influence', 0.0),
                'kickoff_time':    fx.get('kickoff_time', ''),
                'minutes':         stats.get('minutes', 0),
                'modified':        False,
                'opponent_team':   opp_id,
                'own_goals':       stats.get('own_goals', 0),
                'penalties_missed': stats.get('penalties_missed', 0),
                'penalties_saved': stats.get('penalties_saved', 0),
                'recoveries':      stats.get('recoveries', None),
                'red_cards':       stats.get('red_cards', 0),
                'round':           gw,
                'saves':           stats.get('saves', 0),
                'selected':        el.get('selected_by_percent', None),
                'starts':          stats.get('starts', None),
                'tackles':         stats.get('tackles', None),
                'team_a_score':    fx.get('team_a_score', None),
                'team_h_score':    fx.get('team_h_score', None),
                'threat':          stats.get('threat', 0.0),
                'total_points':    stats.get('total_points', 0),
                'transfers_balance': el.get('transfers_balance', 0),
                'transfers_in':    el.get('transfers_in_event', 0),
                'transfers_out':   el.get('transfers_out_event', 0),
                'value':           el.get('now_cost', None),
                'was_home':        is_home,
                'yellow_cards':    stats.get('yellow_cards', 0),
                'GW':              gw,
                # Manager columns (populated for manager rows only)
                'mng_win':         stats.get('mng_win', None),
                'mng_draw':        stats.get('mng_draw', None),
                'mng_loss':        stats.get('mng_loss', None),
                'mng_underdog_win': stats.get('mng_underdog_win', None),
                'mng_underdog_draw': stats.get('mng_underdog_draw', None),
                'mng_clean_sheets': stats.get('mng_clean_sheets', None),
                'mng_goals_scored': stats.get('mng_goals_scored', None),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning(f'[build] GW {gw}: no player rows constructed.')
    else:
        log.info(f'[build] GW {gw}: {len(df):,} player-fixture rows')
    return df


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_season_csvs(
    season_label: str,
    gw: int,
    bootstrap: dict,
    fixtures: list[dict],
    live: dict,
    data_dir: Path = DATA_DIR,
) -> None:
    """
    Write/update the two authoritative CSV files for a season:

      data/{season_label}/gws/merged_gw.csv
          Appends new GW rows (deduplicates on gw + fixture).

      data/{season_label}/players_raw.csv
          Overwrites with latest player snapshot from bootstrap.

    Parameters
    ----------
    season_label : e.g. '2025-26'
    gw           : gameweek number being written
    """
    season_dir = data_dir / season_label
    gws_dir    = season_dir / 'gws'
    gws_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. merged_gw.csv ---
    merged_path = gws_dir / 'merged_gw.csv'
    new_df = build_merged_gw(gw, bootstrap, fixtures, live)

    if merged_path.exists():
        existing = pd.read_csv(merged_path)
        # Drop stale rows for same gw+fixture combination
        if 'GW' in existing.columns and 'fixture' in existing.columns:
            existing = existing[
                ~((existing['GW'] == gw) & (existing['fixture'].isin(new_df['fixture'])))
            ]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(merged_path, index=False)
    log.info(f'[write] {merged_path} ({len(combined):,} total rows)')

    # --- 2. players_raw.csv ---
    players_path = season_dir / 'players_raw.csv'
    players_df   = pd.DataFrame(bootstrap['elements'])
    players_df.to_csv(players_path, index=False)
    log.info(f'[write] {players_path} ({len(players_df):,} players)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the fetch script."""
    p = argparse.ArgumentParser(description='FPL API fetcher — writes updated CSVs to data/')
    p.add_argument('--gw', type=int,
                   help='GW number to fetch (default: current finished GW)')
    p.add_argument('--season', default='2025-26',
                   help='Season label (default: 2025-26)')
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
    )
    args      = _parse_args()
    bootstrap = fetch_bootstrap()
    fixtures  = fetch_fixtures()

    gw = args.gw or get_current_gw(bootstrap)
    live = fetch_gw_live(gw)

    write_season_csvs(
        season_label=args.season,
        gw=gw,
        bootstrap=bootstrap,
        fixtures=fixtures,
        live=live,
    )
    print(f'Done. GW {gw} data written to data/{args.season}/')
