"""
ETL loaders — one function per schema table, in loading order.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .schema import (
    SEASONS, LABEL_TO_ID, SEASON_NAME_TO_ID,
    POSITION_MAP, POSITION_STR_TO_CODE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with UTF-8 / latin-1 fallback."""
    for enc in ('utf-8', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot read {path}")


def _col(df: pd.DataFrame, name: str, default=None) -> pd.Series:
    """Return column if it exists, else a Series of `default`."""
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


def _bool_to_int(series: pd.Series) -> pd.Series:
    """Convert True/False/'True'/'False' → 1/0 integer."""
    return series.map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})


def _insert(df: pd.DataFrame, table: str, conn: sqlite3.Connection) -> int:
    """Bulk-insert DataFrame into table. Returns rows inserted."""
    df.to_sql(table, conn, if_exists='append', index=False, method='multi', chunksize=500)
    return len(df)


# ---------------------------------------------------------------------------
# 1. dim_season
# ---------------------------------------------------------------------------

def load_dim_season(conn: sqlite3.Connection) -> None:
    rows = []
    for sid, label, sy, ey, gws, pos, xp, xg, starts, mng, src in SEASONS:
        rows.append({
            'season_id': sid, 'season_label': label,
            'start_year': sy, 'end_year': ey, 'total_gws': gws,
            'has_position_in_gw': pos, 'has_xp': xp,
            'has_xg_stats': xg, 'has_starts': starts,
            'has_mng_cols': mng, 'team_map_source': src,
        })
    n = _insert(pd.DataFrame(rows), 'dim_season', conn)
    print(f"  dim_season: {n} rows")


# ---------------------------------------------------------------------------
# 2. dim_player
# ---------------------------------------------------------------------------

def load_dim_player(conn: sqlite3.Connection, data_root: Path) -> None:
    frames = []
    for sid, label, *_ in SEASONS:
        path = data_root / label / 'players_raw.csv'
        if not path.exists():
            print(f"  WARNING: missing {path}")
            continue
        df = _read_csv(path)
        df['season_id'] = sid
        frames.append(df)

    union = pd.concat(frames, ignore_index=True)

    # Stable debut season per player_code
    debut = (union.groupby('code')['season_id']
             .min().reset_index()
             .rename(columns={'code': 'player_code', 'season_id': 'debut_season_id'}))

    # Latest-season snapshot per player_code
    latest = (union.sort_values('season_id', ascending=False)
              .drop_duplicates(subset='code')
              .rename(columns={'code': 'player_code'}))

    players = latest.merge(debut, on='player_code')

    out = pd.DataFrame({
        'player_code':      players['player_code'],
        'first_name':       players['first_name'],
        'second_name':      players['second_name'],
        'web_name':         players['web_name'],
        'birth_date':       _col(players, 'birth_date'),
        'region':           _col(players, 'region'),
        'debut_season_id':  players['debut_season_id'],
    })

    n = _insert(out, 'dim_player', conn)
    print(f"  dim_player: {n} rows")


# ---------------------------------------------------------------------------
# 3. dim_team
# ---------------------------------------------------------------------------

def load_dim_team(conn: sqlite3.Connection, data_root: Path) -> None:
    rows = []

    # --- Seasons 1–8: master_team_list.csv provides team_id + team_name ---
    mtl_path = data_root / 'master_team_list.csv'
    mtl = _read_csv(mtl_path)  # columns: season, team, team_name
    mtl.columns = mtl.columns.str.strip()

    for sid, label, *_ in SEASONS:
        if sid > 8:
            continue
        season_mtl = mtl[mtl['season'] == label]
        # Get team_code from players_raw for this season
        raw_path = data_root / label / 'players_raw.csv'
        raw = _read_csv(raw_path, usecols=['team', 'team_code'])
        team_code_map = (raw.dropna(subset=['team'])
                         .drop_duplicates('team')
                         .set_index('team')['team_code']
                         .to_dict())
        for _, r in season_mtl.iterrows():
            tid = int(r['team'])
            rows.append({
                'season_id': sid,
                'team_id':   tid,
                'team_name': r['team_name'],
                'team_code': int(team_code_map[tid]) if tid in team_code_map and pd.notna(team_code_map[tid]) else None,
            })

    # --- Seasons 9–10: derive from players_raw + merged_gw ---
    for sid, label, *_ in SEASONS:
        if sid <= 8:
            continue
        raw = _read_csv(data_root / label / 'players_raw.csv', usecols=['id', 'team', 'team_code'])
        gw  = _read_csv(data_root / label / 'gws' / 'merged_gw.csv', usecols=['element', 'team'])
        gw  = gw.rename(columns={'team': 'team_name_str'})

        merged = raw.merge(
            gw[['element', 'team_name_str']].drop_duplicates('element'),
            left_on='id', right_on='element', how='left'
        )
        # Mode team_name per team_id
        team_names = (merged.dropna(subset=['team_name_str'])
                      .groupby('team')['team_name_str']
                      .agg(lambda x: x.mode().iloc[0]))
        team_codes = raw.dropna(subset=['team']).drop_duplicates('team').set_index('team')['team_code']

        for tid, tname in team_names.items():
            tc = team_codes.get(tid)
            rows.append({
                'season_id': sid,
                'team_id':   int(tid),
                'team_name': tname,
                'team_code': int(tc) if pd.notna(tc) else None,
            })

    n = _insert(pd.DataFrame(rows), 'dim_team', conn)
    print(f"  dim_team: {n} rows")


# ---------------------------------------------------------------------------
# 4. dim_player_season
# ---------------------------------------------------------------------------

def load_dim_player_season(
    conn: sqlite3.Connection,
    data_root: Path,
    history_costs: dict,  # (player_code, season_id) → {'start_cost', 'end_cost'}
) -> None:
    # Build lookup: (season_id, team_id) → team_sk
    team_sk_df = pd.read_sql("SELECT season_id, team_id, team_sk FROM dim_team", conn)
    team_sk_map = team_sk_df.set_index(['season_id', 'team_id'])['team_sk'].to_dict()

    total = 0
    for sid, label, *_ in SEASONS:
        path = data_root / label / 'players_raw.csv'
        if not path.exists():
            continue
        raw = _read_csv(path)

        # element_type: 1=GK,2=DEF,3=MID,4=FWD,5=AM→MID. Map to stable int code.
        _code_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 3}
        pos_code = pd.to_numeric(raw['element_type'], errors='coerce').map(_code_map).fillna(3).astype(int)

        # start_cost: derived from players_raw; override with history if available
        derived_start = raw['now_cost'] - raw['cost_change_start']

        start_costs, end_costs = [], []
        for _, r in raw.iterrows():
            key = (int(r['code']), sid)
            hc = history_costs.get(key, {})
            start_costs.append(hc.get('start_cost') or (int(derived_start[r.name]) if pd.notna(derived_start[r.name]) else None))
            end_costs.append(hc.get('end_cost'))

        team_ids = raw['team'].fillna(-1).astype(int)
        team_sks = [team_sk_map.get((sid, tid)) for tid in team_ids]

        out = pd.DataFrame({
            'season_id':           sid,
            'player_code':         raw['code'].astype(int),
            'fpl_id':              raw['id'].astype(int),
            'team_sk':             team_sks,
            'position_code':       pos_code,
            'position_label':      pos_code.map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}),
            'start_cost':          start_costs,
            'end_cost':            end_costs,
            'status':              _col(raw, 'status'),
            'selected_by_percent': pd.to_numeric(_col(raw, 'selected_by_percent'), errors='coerce'),
            'total_points':        pd.to_numeric(_col(raw, 'total_points'), errors='coerce'),
            'minutes':             pd.to_numeric(_col(raw, 'minutes'), errors='coerce'),
            'goals_scored':        pd.to_numeric(_col(raw, 'goals_scored'), errors='coerce'),
            'assists':             pd.to_numeric(_col(raw, 'assists'), errors='coerce'),
            'clean_sheets':        pd.to_numeric(_col(raw, 'clean_sheets'), errors='coerce'),
            'saves':               pd.to_numeric(_col(raw, 'saves'), errors='coerce'),
            'bonus':               pd.to_numeric(_col(raw, 'bonus'), errors='coerce'),
            'bps':                 pd.to_numeric(_col(raw, 'bps'), errors='coerce'),
            'yellow_cards':        pd.to_numeric(_col(raw, 'yellow_cards'), errors='coerce'),
            'red_cards':           pd.to_numeric(_col(raw, 'red_cards'), errors='coerce'),
            'transfers_in':        pd.to_numeric(_col(raw, 'transfers_in'), errors='coerce'),
            'transfers_out':       pd.to_numeric(_col(raw, 'transfers_out'), errors='coerce'),
            'ict_index':           pd.to_numeric(_col(raw, 'ict_index'), errors='coerce'),
            'influence':           pd.to_numeric(_col(raw, 'influence'), errors='coerce'),
            'creativity':          pd.to_numeric(_col(raw, 'creativity'), errors='coerce'),
            'threat':              pd.to_numeric(_col(raw, 'threat'), errors='coerce'),
            'season_xg':           pd.to_numeric(_col(raw, 'expected_goals'), errors='coerce'),
            'season_xa':           pd.to_numeric(_col(raw, 'expected_assists'), errors='coerce'),
            'season_xgi':          pd.to_numeric(_col(raw, 'expected_goal_involvements'), errors='coerce'),
            'season_xgc':          pd.to_numeric(_col(raw, 'expected_goals_conceded'), errors='coerce'),
        })

        n = _insert(out, 'dim_player_season', conn)
        total += n

    print(f"  dim_player_season: {total} rows")


# ---------------------------------------------------------------------------
# 5. fact_player_season_history  (also returns cost index for dim_player_season)
# ---------------------------------------------------------------------------

def scan_history(data_root: Path) -> pd.DataFrame:
    """
    Read all players/*/history.csv files across all seasons.
    Deduplicate on (element_code, season_name), keeping the latest source season.
    Returns the combined DataFrame (used by both load functions below).
    """
    records = []
    for sid, label, *_ in SEASONS:
        players_dir = data_root / label / 'players'
        if not players_dir.exists():
            continue
        for player_dir in players_dir.iterdir():
            hist_path = player_dir / 'history.csv'
            if not hist_path.exists():
                continue
            try:
                df = _read_csv(hist_path)
                df['source_season_id'] = sid
                records.append(df)
            except Exception as e:
                print(f"    Warning: {hist_path.name}: {e}")

    if not records:
        return pd.DataFrame()

    all_hist = pd.concat(records, ignore_index=True)

    # Deduplicate: keep row from latest source file per (element_code, season_name)
    all_hist = (all_hist
                .sort_values('source_season_id', ascending=False)
                .drop_duplicates(subset=['element_code', 'season_name'])
                .reset_index(drop=True))

    return all_hist


def build_history_cost_index(history_df: pd.DataFrame) -> dict:
    """
    Returns {(player_code, season_id): {'start_cost': int, 'end_cost': int}}
    for use in dim_player_season loading.
    """
    if history_df.empty:
        return {}
    index = {}
    for _, r in history_df.iterrows():
        sid = SEASON_NAME_TO_ID.get(str(r.get('season_name', '')))
        if sid is None:
            continue
        pc = r.get('element_code')
        if pd.isna(pc):
            continue
        sc = r.get('start_cost')
        ec = r.get('end_cost')
        index[(int(pc), sid)] = {
            'start_cost': int(sc) if pd.notna(sc) else None,
            'end_cost':   int(ec) if pd.notna(ec) else None,
        }
    return index


def load_fact_player_season_history(
    conn: sqlite3.Connection,
    history_df: pd.DataFrame,
) -> None:
    if history_df.empty:
        print("  fact_player_season_history: 0 rows (no history files found)")
        return

    # Load valid player_codes and season_ids from DB for FK filtering
    valid_codes   = set(pd.read_sql("SELECT player_code FROM dim_player", conn)['player_code'])
    valid_seasons = set(SEASON_NAME_TO_ID.values())

    df = history_df.copy()
    df['season_id']   = df['season_name'].map(SEASON_NAME_TO_ID)
    df['player_code'] = df['element_code'].astype('Int64')

    # Filter to valid FKs only
    df = df[df['player_code'].isin(valid_codes) & df['season_id'].isin(valid_seasons)]

    out = pd.DataFrame({
        'player_code':      df['player_code'],
        'season_id':        df['season_id'],
        'start_cost':       pd.to_numeric(df.get('start_cost'),       errors='coerce'),
        'end_cost':         pd.to_numeric(df.get('end_cost'),         errors='coerce'),
        'total_points':     pd.to_numeric(df.get('total_points'),     errors='coerce'),
        'minutes':          pd.to_numeric(df.get('minutes'),          errors='coerce'),
        'goals_scored':     pd.to_numeric(df.get('goals_scored'),     errors='coerce'),
        'assists':          pd.to_numeric(df.get('assists'),          errors='coerce'),
        'clean_sheets':     pd.to_numeric(df.get('clean_sheets'),     errors='coerce'),
        'saves':            pd.to_numeric(df.get('saves'),            errors='coerce'),
        'bonus':            pd.to_numeric(df.get('bonus'),            errors='coerce'),
        'bps':              pd.to_numeric(df.get('bps'),              errors='coerce'),
        'yellow_cards':     pd.to_numeric(df.get('yellow_cards'),     errors='coerce'),
        'red_cards':        pd.to_numeric(df.get('red_cards'),        errors='coerce'),
        'own_goals':        pd.to_numeric(df.get('own_goals'),        errors='coerce'),
        'penalties_missed': pd.to_numeric(df.get('penalties_missed'), errors='coerce'),
        'penalties_saved':  pd.to_numeric(df.get('penalties_saved'),  errors='coerce'),
        'ict_index':        pd.to_numeric(df.get('ict_index'),        errors='coerce'),
        'influence':        pd.to_numeric(df.get('influence'),        errors='coerce'),
        'creativity':       pd.to_numeric(df.get('creativity'),       errors='coerce'),
        'threat':           pd.to_numeric(df.get('threat'),           errors='coerce'),
    })

    n = _insert(out, 'fact_player_season_history', conn)
    print(f"  fact_player_season_history: {n} rows")


# ---------------------------------------------------------------------------
# 6. fact_gw_player
# ---------------------------------------------------------------------------

# All era-specific columns that may or may not appear in merged_gw.csv
_OPTIONAL_GW_COLS = [
    'starts',
    'xP',  # renamed → xp
    'expected_goals', 'expected_assists',
    'expected_goal_involvements', 'expected_goals_conceded',
    'modified',
    'big_chances_created', 'big_chances_missed', 'key_passes',
    'completed_passes', 'attempted_passes', 'dribbles', 'fouls', 'offside',
    'errors_leading_to_goal', 'errors_leading_to_goal_attempt',
    'open_play_crosses', 'target_missed', 'winning_goals',
    'clearances_blocks_interceptions', 'recoveries', 'tackles',
    'defensive_contribution',
    'mng_win', 'mng_draw', 'mng_loss', 'mng_goals_scored',
    'mng_clean_sheets', 'mng_underdog_win', 'mng_underdog_draw',
]

# Counting stats that must never be NULL in the DB (NOT NULL DEFAULT 0)
_COUNTING_COLS = [
    'minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets',
    'goals_conceded', 'own_goals', 'penalties_missed', 'penalties_saved',
    'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps',
]


def load_fact_gw_player(conn: sqlite3.Connection, data_root: Path) -> None:
    # Build lookup tables from already-loaded dims
    team_sk_df = pd.read_sql("SELECT season_id, team_id, team_sk FROM dim_team", conn)
    team_sk_map = team_sk_df.set_index(['season_id', 'team_id'])['team_sk'].to_dict()

    total = 0
    for sid, label, *_ in SEASONS:
        gw_path = data_root / label / 'gws' / 'merged_gw.csv'
        raw_path = data_root / label / 'players_raw.csv'
        if not gw_path.exists():
            print(f"  WARNING: missing {gw_path}")
            continue

        gw  = _read_csv(gw_path)
        raw = _read_csv(raw_path, usecols=['id', 'code'])

        # ── player_code bridge ───────────────────────────────────────────────
        bridge = raw.set_index('id')['code'].to_dict()
        gw['player_code'] = gw['element'].map(bridge)
        missing = gw['player_code'].isna()
        if missing.any():
            print(f"    {label}: {missing.sum()} rows skipped (element not in players_raw)")
        gw = gw[~missing].copy()
        gw['player_code'] = gw['player_code'].astype(int)

        # ── team lookups ─────────────────────────────────────────────────────
        # opponent team_sk via (season_id, opponent_team numeric id)
        gw['opponent_team_sk'] = gw['opponent_team'].apply(
            lambda tid: team_sk_map.get((sid, int(tid))) if pd.notna(tid) else None
        )

        # player's own team_sk from dim_player_season (static end-of-season assignment)
        ps = pd.read_sql(
            f"SELECT player_code, team_sk, position_code, position_label "
            f"FROM dim_player_season WHERE season_id = {sid}",
            conn,
        ).set_index('player_code')
        gw['team_sk'] = gw['player_code'].map(ps['team_sk'])

        # ── position ─────────────────────────────────────────────────────────
        if 'position' in gw.columns:
            gw['position_code']  = gw['position'].map(POSITION_STR_TO_CODE)
            gw['position_label'] = gw['position']
        else:
            gw['position_code']  = gw['player_code'].map(ps['position_code'])
            gw['position_label'] = gw['player_code'].map(ps['position_label'])

        # ── booleans ─────────────────────────────────────────────────────────
        gw['was_home'] = _bool_to_int(gw['was_home'])
        if 'modified' in gw.columns:
            gw['modified'] = _bool_to_int(gw['modified'])

        # ── build output dict, then construct DataFrame in one shot ──────────
        data: dict = {
            'season_id':        pd.Series([sid] * len(gw), index=gw.index),
            'gw':               gw['GW'],
            'fixture_id':       pd.to_numeric(gw['fixture'], errors='coerce'),
            'player_code':      gw['player_code'],
            'fpl_id':           gw['element'].astype(int),
            'team_sk':          gw['team_sk'],
            'opponent_team_sk': gw['opponent_team_sk'],
            'position_code':    gw['position_code'],
            'position_label':   gw['position_label'],
            'kickoff_time':     gw['kickoff_time'],
            'was_home':         gw['was_home'],
            'value':            pd.to_numeric(gw['value'], errors='coerce'),
        }

        # Core counting stats (NOT NULL DEFAULT 0)
        for col in _COUNTING_COLS:
            data[col] = pd.to_numeric(_col(gw, col, 0), errors='coerce').fillna(0).astype(int)

        # ICT
        for col in ('ict_index', 'influence', 'creativity', 'threat'):
            data[col] = pd.to_numeric(_col(gw, col), errors='coerce')

        # Market
        for col in ('selected', 'transfers_in', 'transfers_out', 'transfers_balance',
                    'team_h_score', 'team_a_score'):
            data[col] = pd.to_numeric(_col(gw, col), errors='coerce')

        # Optional / era-specific columns
        for src_col in _OPTIONAL_GW_COLS:
            if src_col not in gw.columns:
                continue
            db_col = 'xp' if src_col == 'xP' else src_col
            data[db_col] = pd.to_numeric(gw[src_col], errors='coerce')

        out = pd.DataFrame(data)

        before = len(out)
        out = out.drop_duplicates(subset=['season_id', 'gw', 'fpl_id', 'fixture_id'])
        dupes = before - len(out)
        if dupes:
            print(f"    {label}: dropped {dupes} duplicate rows")
        n = _insert(out, 'fact_gw_player', conn)
        total += n
        print(f"    {label}: {n} rows")

    print(f"  fact_gw_player: {total} rows total")
