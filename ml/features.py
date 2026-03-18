"""
Feature engineering for FPL GW-level points prediction.

Entry point: build_feature_matrix(position, era='xg')

Builds a feature matrix for one position from db/fpl.db applying:
  - Base filter (section 4.1 of docs/project_plan.md): xG era (seasons 7-10),
    mng_win IS NULL, minutes > 0, position_label filtered, season_gw_count >= 5
  - Player rolling features within (player_code, season_id) only
  - Team match-level features derived from score columns (not goals_conceded,
    which reflects only time on pitch and is inconsistent across players)
  - Opponent season rank derived from end-of-season standings per season
  - Lag features for value and transfer activity
  - Position-specific feature selection per section 4.6 of docs/project_plan.md

Banned features (leakage): bonus/bps/ict_index (same-GW post-match);
clean_sheets/goals_scored/assists as same-GW values (use rolling lags only);
transfers_in/out same-GW (lag by 1). See section 4.4 of docs/project_plan.md.

Caches to outputs/features/feature_matrix_{position}.parquet.
"""

import os
import sqlite3

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, '..', 'db', 'fpl.db')
OUTPUTS_DIR = os.path.join(_HERE, '..', 'outputs', 'features')

XG_ERA_MIN_SEASON = 7  # 2022-23
MIN_GW_COUNT = 5       # minimum qualifying GW appearances per player-season

VALID_POSITIONS = ('GK', 'DEF', 'MID', 'FWD')

# Columns always included in the output (identity/grouping, not predictive)
CONTEXT_COLS = ['season_id', 'gw', 'fixture_id', 'player_code', 'position_code', 'team_sk']
TARGET_COL = 'total_points'

# Per-position feature lists (sections 4.5 and 4.6 of docs/project_plan.md).
# team_goals_conceded_season and team_cs_rolling_3gw are included for GK/DEF only
# (explains 46.6% variance in DEF/GK scoring).
# xG/xA/xGI features are MID/FWD only.
# saves_rolling_5gw is GK only.
_POSITION_FEATURES = {
    'GK': [
        'was_home',
        'opponent_season_rank',
        'team_goals_conceded_season',
        'team_cs_rolling_3gw',
        'team_goals_scored_rolling_3gw',
        'pts_rolling_3gw',
        'pts_rolling_5gw',
        'mins_rolling_3gw',
        'cs_rolling_5gw',
        'saves_rolling_5gw',
        'bonus_rolling_5gw',
        'xgc_rolling_5gw',
        'season_pts_per_gw_to_date',
        'season_starts_rate_to_date',
        'start_cost',
        'value_lag1',
        'transfers_in_lag1',
        'transfers_out_lag1',
        'opponent_goals_scored_season',
        'opponent_cs_rate_season',
    ],
    'DEF': [
        'was_home',
        'opponent_season_rank',
        'team_goals_conceded_season',
        'team_cs_rolling_3gw',
        'team_goals_scored_rolling_3gw',
        'pts_rolling_3gw',
        'pts_rolling_5gw',
        'mins_rolling_3gw',
        'cs_rolling_5gw',
        'bonus_rolling_5gw',
        'xgc_rolling_5gw',
        'season_pts_per_gw_to_date',
        'season_starts_rate_to_date',
        'start_cost',
        'value_lag1',
        'transfers_in_lag1',
        'transfers_out_lag1',
        'opponent_goals_scored_season',
        'opponent_cs_rate_season',
    ],
    'MID': [
        'was_home',
        'opponent_season_rank',
        'team_goals_scored_rolling_3gw',
        'pts_rolling_3gw',
        'pts_rolling_5gw',
        'mins_rolling_3gw',
        'goals_rolling_5gw',
        'assists_rolling_5gw',
        'bonus_rolling_5gw',
        'xg_rolling_5gw',
        'xa_rolling_5gw',
        'xgi_rolling_5gw',
        'season_pts_per_gw_to_date',
        'season_starts_rate_to_date',
        'start_cost',
        'value_lag1',
        'transfers_in_lag1',
        'transfers_out_lag1',
        'opponent_goals_scored_season',
        'opponent_cs_rate_season',
    ],
    'FWD': [
        'was_home',
        'opponent_season_rank',
        'team_goals_scored_rolling_3gw',
        'pts_rolling_3gw',
        'pts_rolling_5gw',
        'mins_rolling_3gw',
        'goals_rolling_5gw',
        'assists_rolling_5gw',
        'bonus_rolling_5gw',
        'xg_rolling_5gw',
        'xa_rolling_5gw',
        'xgi_rolling_5gw',
        'season_pts_per_gw_to_date',
        'season_starts_rate_to_date',
        'start_cost',
        'value_lag1',
        'transfers_in_lag1',
        'transfers_out_lag1',
        'opponent_goals_scored_season',
        'opponent_cs_rate_season',
    ],
}

# CTE that derives end-of-season league rank per team per season.
# Rank 1 = most points (champion), rank 20 = fewest points (bottom).
# For the current season (2025-26), this uses partial-season standings.
_OPP_RANK_CTE = """
    _fix AS (
        SELECT DISTINCT season_id, gw, fixture_id, team_sk, was_home,
            team_h_score, team_a_score
        FROM fact_gw_player
        WHERE team_h_score IS NOT NULL AND team_a_score IS NOT NULL
          AND mng_win IS NULL
    ),
    _match_pts AS (
        SELECT
            season_id,
            team_sk,
            CASE
                WHEN was_home = 1 AND team_h_score > team_a_score THEN 3
                WHEN was_home = 0 AND team_a_score > team_h_score THEN 3
                WHEN team_h_score = team_a_score THEN 1
                ELSE 0
            END AS pts,
            CASE WHEN was_home = 1 THEN team_h_score ELSE team_a_score END AS gf,
            CASE WHEN was_home = 1 THEN team_a_score ELSE team_h_score END AS ga
        FROM _fix
    ),
    _season_table AS (
        SELECT season_id, team_sk,
            SUM(pts)         AS total_pts,
            SUM(gf) - SUM(ga) AS gd,
            SUM(gf)          AS gf_total
        FROM _match_pts
        GROUP BY season_id, team_sk
    ),
    _opp_rank AS (
        SELECT season_id, team_sk,
            RANK() OVER (
                PARTITION BY season_id
                ORDER BY total_pts DESC, gd DESC, gf_total DESC
            ) AS season_rank
        FROM _season_table
    )"""

# Primary data query: one row per player per fixture, xG era, position-filtered.
# opponent_season_rank joined via opponent_team_sk.
# goals_conceded is excluded here; team-level conceded derived from scores in Python.
_BASE_SQL = f"""
WITH {_OPP_RANK_CTE}
SELECT
    fgp.season_id,
    fgp.gw,
    fgp.fixture_id,
    fgp.player_code,
    fgp.team_sk,
    fgp.opponent_team_sk,
    fgp.was_home,
    fgp.minutes,
    fgp.starts,
    fgp.total_points,
    fgp.goals_scored,
    fgp.assists,
    fgp.clean_sheets,
    fgp.saves,
    fgp.bonus,
    fgp.value,
    fgp.transfers_in,
    fgp.transfers_out,
    fgp.expected_goals,
    fgp.expected_assists,
    fgp.expected_goal_involvements,
    fgp.expected_goals_conceded,
    fgp.team_h_score,
    fgp.team_a_score,
    dps.start_cost,
    dps.position_code,
    COALESCE(_opp_rank.season_rank, 10) AS opponent_season_rank
FROM fact_gw_player fgp
JOIN dim_player_season dps
    ON fgp.season_id = dps.season_id AND fgp.player_code = dps.player_code
LEFT JOIN _opp_rank
    ON fgp.opponent_team_sk = _opp_rank.team_sk
   AND fgp.season_id = _opp_rank.season_id
WHERE fgp.mng_win IS NULL
  AND fgp.minutes > 0
  AND fgp.position_label = ?
  AND fgp.season_id >= ?
ORDER BY fgp.player_code, fgp.season_id, fgp.gw, fgp.fixture_id
"""

# Team match data query: one row per team per fixture.
# Uses score columns (not goals_conceded) since goals_conceded in fact_gw_player
# reflects only goals conceded while a specific player was on the pitch and is
# inconsistent across players in the same fixture.
_TEAM_MATCH_SQL = """
SELECT
    season_id, gw, fixture_id, team_sk,
    MIN(was_home)     AS was_home,
    MIN(team_h_score) AS team_h_score,
    MIN(team_a_score) AS team_a_score
FROM fact_gw_player
WHERE mng_win IS NULL
  AND team_h_score IS NOT NULL
  AND season_id >= ?
GROUP BY season_id, gw, fixture_id, team_sk
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    position: str,
    era: str = 'xg',
    force: bool = False,
) -> pd.DataFrame:
    """
    Build ML-ready feature matrix for one position.

    Parameters
    ----------
    position : str   One of 'GK', 'DEF', 'MID', 'FWD'.
    era      : str   Must be 'xg' (seasons 7-10, 2022-23 to 2025-26).
    force    : bool  Recompute even if a cached parquet exists.

    Returns
    -------
    pd.DataFrame
        One row per player-fixture with CONTEXT_COLS, total_points (target),
        and position-specific features from section 4.5/4.6 of docs/project_plan.md.
        Cached to outputs/features/feature_matrix_{position}.parquet.

    Notes
    -----
    Rolling features are computed within (player_code, season_id) only.
    No chaining across season boundaries (section 4.4 rolling boundary rule).
    For double gameweeks the second fixture's rolling features include the first;
    this is minor since both fixtures are played within the same GW window.
    """
    if position not in VALID_POSITIONS:
        raise ValueError(f"position must be one of {VALID_POSITIONS}, got '{position}'")
    if era != 'xg':
        raise ValueError("Only era='xg' is supported")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    cache_path = os.path.join(OUTPUTS_DIR, f'feature_matrix_{position}.parquet')

    if not force and os.path.exists(cache_path):
        print(f"[features] Loading cached {position} matrix from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[features] Building {position} feature matrix ...")

    df = _load_base_data(position)
    df = _apply_gw_count_filter(df)
    df = _add_player_rolling_features(df)
    df = _add_season_to_date_features(df)
    df = _add_lag_features(df)

    tm = _load_team_match_data()
    df = _add_team_features(df, tm)
    df = _add_opponent_features(df, tm)

    df = _select_position_features(df, position)
    df.to_parquet(cache_path, index=False)

    print(f"[features] {position}: {len(df):,} rows x {len(df.columns)} cols -> {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_base_data(position: str) -> pd.DataFrame:
    """Query fpl.db for one position's xG era rows using the base SQL + opponent rank CTE."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(_BASE_SQL, conn, params=(position, XG_ERA_MIN_SEASON))
    conn.close()
    return df


def _load_team_match_data() -> pd.DataFrame:
    """One row per team per fixture (all positions combined, xG era)."""
    conn = sqlite3.connect(DB_PATH)
    tm = pd.read_sql_query(_TEAM_MATCH_SQL, conn, params=(XG_ERA_MIN_SEASON,))
    conn.close()
    tm['team_gf'] = np.where(tm['was_home'] == 1, tm['team_h_score'], tm['team_a_score'])
    tm['team_ga'] = np.where(tm['was_home'] == 1, tm['team_a_score'], tm['team_h_score'])
    return (
        tm[['season_id', 'gw', 'fixture_id', 'team_sk', 'team_gf', 'team_ga']]
        .sort_values(['team_sk', 'season_id', 'gw', 'fixture_id'])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _apply_gw_count_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop player-seasons with fewer than MIN_GW_COUNT qualifying appearances."""
    counts = df.groupby(['player_code', 'season_id'])['gw'].transform('count')
    return df[counts >= MIN_GW_COUNT].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player-level rolling and lag features
# ---------------------------------------------------------------------------

def _add_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling means of prior GWs within (player_code, season_id).
    shift(1) ensures the current GW is excluded (no same-GW leakage).
    min_periods=1 avoids NaN for early appearances (2nd GW onward has a value).
    First GW of each player-season always produces NaN (no prior data available).
    """
    g = df.groupby(['player_code', 'season_id'], sort=False)

    def roll(col: str, window: int) -> pd.Series:
        return g[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    df['pts_rolling_3gw']   = roll('total_points', 3)
    df['pts_rolling_5gw']   = roll('total_points', 5)
    df['mins_rolling_3gw']  = roll('minutes', 3)
    df['goals_rolling_5gw']    = roll('goals_scored', 5)
    df['assists_rolling_5gw']  = roll('assists', 5)
    df['cs_rolling_5gw']       = roll('clean_sheets', 5)
    df['bonus_rolling_5gw']    = roll('bonus', 5)
    df['saves_rolling_5gw']    = roll('saves', 5)
    df['xg_rolling_5gw']   = roll('expected_goals', 5)
    df['xa_rolling_5gw']   = roll('expected_assists', 5)
    df['xgi_rolling_5gw']  = roll('expected_goal_involvements', 5)
    df['xgc_rolling_5gw']  = roll('expected_goals_conceded', 5)
    return df


def _add_season_to_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative per-player season averages using only prior GW data."""
    g = df.groupby(['player_code', 'season_id'], sort=False)
    df['season_pts_per_gw_to_date'] = g['total_points'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df['season_starts_rate_to_date'] = g['starts'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single-GW lag features within (player_code, season_id)."""
    g = df.groupby(['player_code', 'season_id'], sort=False)
    df['value_lag1']         = g['value'].transform(lambda x: x.shift(1))
    df['transfers_in_lag1']  = g['transfers_in'].transform(lambda x: x.shift(1))
    df['transfers_out_lag1'] = g['transfers_out'].transform(lambda x: x.shift(1))
    return df


# ---------------------------------------------------------------------------
# Team-level features (computed from team match data, not player rows)
# ---------------------------------------------------------------------------

def _add_team_features(df: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """
    Team match-level rolling and cumulative features for the player's own team.
    All features use prior-match data only (shift 1 before rolling/expanding).

    team_goals_conceded_season : cumulative goals conceded by team up to this fixture
    team_cs_rolling_3gw        : proportion of last 3 matches the team kept a CS
    team_goals_scored_rolling_3gw : rolling mean goals scored over last 3 matches
    """
    tm = tm.copy()
    tg = tm.groupby(['team_sk', 'season_id'], sort=False)

    tm['team_goals_conceded_season'] = tg['team_ga'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).sum()
    )
    tm['_cs'] = (tm['team_ga'] == 0).astype(float)
    tm['team_cs_rolling_3gw'] = tg['_cs'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    tm['team_goals_scored_rolling_3gw'] = tg['team_gf'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    team_cols = ['season_id', 'gw', 'fixture_id', 'team_sk',
                 'team_goals_conceded_season', 'team_cs_rolling_3gw',
                 'team_goals_scored_rolling_3gw']
    df = df.merge(tm[team_cols], on=['season_id', 'gw', 'fixture_id', 'team_sk'], how='left')
    return df


def _add_opponent_features(df: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """
    Opponent team's season-to-date attacking and defensive stats.
    Joined via opponent_team_sk.

    opponent_goals_scored_season : cumulative goals scored by opponent before this fixture
    opponent_cs_rate_season      : opponent's clean sheet rate before this fixture
    """
    tm = tm.copy()
    tg = tm.groupby(['team_sk', 'season_id'], sort=False)

    tm['_cumul_gf'] = tg['team_gf'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).sum()
    )
    # (NaN == 0) evaluates to False, so _cs_flag treats first-row NaN as 0
    tm['_cs_flag'] = (tm['team_ga'] == 0).astype(float)
    tm['_cumul_cs'] = tg['_cs_flag'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).sum()
    )
    # cumcount() = 0-indexed row position within group = number of prior matches
    tm['_prior_matches'] = tg.cumcount()

    tm['opponent_goals_scored_season'] = tm['_cumul_gf']
    tm['opponent_cs_rate_season'] = (
        tm['_cumul_cs'] / tm['_prior_matches'].replace(0, np.nan)
    )

    opp_cols = (
        tm[['season_id', 'gw', 'fixture_id', 'team_sk',
            'opponent_goals_scored_season', 'opponent_cs_rate_season']]
        .rename(columns={'team_sk': 'opponent_team_sk'})
    )
    df = df.merge(opp_cols, on=['season_id', 'gw', 'fixture_id', 'opponent_team_sk'], how='left')
    return df


# ---------------------------------------------------------------------------
# Output selection
# ---------------------------------------------------------------------------

def _select_position_features(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Retain context columns, target, and position-specific feature columns."""
    feature_cols = _POSITION_FEATURES[position]
    output_cols = CONTEXT_COLS + [TARGET_COL] + feature_cols
    # Guard against any column missing due to unexpected schema gaps
    output_cols = [c for c in output_cols if c in df.columns]
    return df[output_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI: run all four positions and print summary
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for pos in VALID_POSITIONS:
        matrix = build_feature_matrix(pos, force=True)
        nan_rates = matrix.isnull().mean().sort_values(ascending=False)
        print(f"\n{pos}: {len(matrix):,} rows x {len(matrix.columns)} columns")
        nonzero_nan = nan_rates[nan_rates > 0]
        if nonzero_nan.empty:
            print("  No NaN values.")
        else:
            print("  NaN rates (non-zero):")
            for col, rate in nonzero_nan.items():
                print(f"    {col:<35} {rate:.1%}")
