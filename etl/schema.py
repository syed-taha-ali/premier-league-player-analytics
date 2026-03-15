"""
Schema constants: season seed data, lookup maps, and DDL.
"""

# (season_id, label, start_year, end_year, total_gws,
#  has_position_in_gw, has_xp, has_xg_stats, has_starts, has_mng_cols,
#  team_map_source)
SEASONS = [
    (1,  '2016-17', 2016, 2017, 38, 0, 0, 0, 0, 0, 'master_team_list'),
    (2,  '2017-18', 2017, 2018, 38, 0, 0, 0, 0, 0, 'master_team_list'),
    (3,  '2018-19', 2018, 2019, 38, 0, 0, 0, 0, 0, 'master_team_list'),
    (4,  '2019-20', 2019, 2020, 47, 0, 0, 0, 0, 0, 'master_team_list'),
    (5,  '2020-21', 2020, 2021, 38, 1, 1, 0, 0, 0, 'master_team_list'),
    (6,  '2021-22', 2021, 2022, 38, 1, 1, 0, 0, 0, 'master_team_list'),
    (7,  '2022-23', 2022, 2023, 38, 1, 1, 1, 1, 0, 'master_team_list'),
    (8,  '2023-24', 2023, 2024, 38, 1, 1, 1, 1, 0, 'master_team_list'),
    (9,  '2024-25', 2024, 2025, 38, 1, 1, 1, 1, 1, 'derived'),
    (10, '2025-26', 2025, 2026, 38, 1, 1, 1, 0, 0, 'derived'),
]

LABEL_TO_ID = {s[1]: s[0] for s in SEASONS}

# Maps history.csv season_name field ("2021/22") → season_id
SEASON_NAME_TO_ID = {
    '2016/17': 1, '2017/18': 2, '2018/19': 3, '2019/20': 4,
    '2020/21': 5, '2021/22': 6, '2022/23': 7, '2023/24': 8,
    '2024/25': 9, '2025/26': 10,
}

# players_raw.element_type → position label (5=AM treated as MID)
POSITION_MAP = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD', 5: 'MID'}

# merged_gw.position string → position code
POSITION_STR_TO_CODE = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dim_season (
    season_id           INTEGER PRIMARY KEY,
    season_label        TEXT    NOT NULL UNIQUE,
    start_year          INTEGER NOT NULL,
    end_year            INTEGER NOT NULL,
    total_gws           INTEGER NOT NULL,
    has_position_in_gw  INTEGER NOT NULL DEFAULT 0,
    has_xp              INTEGER NOT NULL DEFAULT 0,
    has_xg_stats        INTEGER NOT NULL DEFAULT 0,
    has_starts          INTEGER NOT NULL DEFAULT 0,
    has_mng_cols        INTEGER NOT NULL DEFAULT 0,
    team_map_source     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_player (
    player_code     INTEGER PRIMARY KEY,
    first_name      TEXT    NOT NULL,
    second_name     TEXT    NOT NULL,
    web_name        TEXT    NOT NULL,
    birth_date      TEXT,
    region          INTEGER,
    debut_season_id INTEGER NOT NULL REFERENCES dim_season(season_id)
);

CREATE TABLE IF NOT EXISTS dim_team (
    team_sk     INTEGER PRIMARY KEY,
    season_id   INTEGER NOT NULL REFERENCES dim_season(season_id),
    team_id     INTEGER NOT NULL,
    team_name   TEXT    NOT NULL,
    team_code   INTEGER,
    UNIQUE (season_id, team_id)
);

CREATE TABLE IF NOT EXISTS dim_player_season (
    season_id           INTEGER NOT NULL REFERENCES dim_season(season_id),
    player_code         INTEGER NOT NULL REFERENCES dim_player(player_code),
    fpl_id              INTEGER NOT NULL,
    team_sk             INTEGER REFERENCES dim_team(team_sk),
    position_code       INTEGER NOT NULL,
    position_label      TEXT    NOT NULL,
    start_cost          INTEGER,
    end_cost            INTEGER,
    status              TEXT,
    selected_by_percent REAL,
    total_points        INTEGER,
    minutes             INTEGER,
    goals_scored        INTEGER,
    assists             INTEGER,
    clean_sheets        INTEGER,
    saves               INTEGER,
    bonus               INTEGER,
    bps                 INTEGER,
    yellow_cards        INTEGER,
    red_cards           INTEGER,
    transfers_in        INTEGER,
    transfers_out       INTEGER,
    ict_index           REAL,
    influence           REAL,
    creativity          REAL,
    threat              REAL,
    season_xg           REAL,
    season_xa           REAL,
    season_xgi          REAL,
    season_xgc          REAL,
    PRIMARY KEY (season_id, player_code),
    UNIQUE (season_id, fpl_id)
);

CREATE TABLE IF NOT EXISTS fact_player_season_history (
    history_sk       INTEGER PRIMARY KEY,
    player_code      INTEGER NOT NULL REFERENCES dim_player(player_code),
    season_id        INTEGER NOT NULL REFERENCES dim_season(season_id),
    start_cost       INTEGER,
    end_cost         INTEGER,
    total_points     INTEGER,
    minutes          INTEGER,
    goals_scored     INTEGER,
    assists          INTEGER,
    clean_sheets     INTEGER,
    saves            INTEGER,
    bonus            INTEGER,
    bps              INTEGER,
    yellow_cards     INTEGER,
    red_cards        INTEGER,
    own_goals        INTEGER,
    penalties_missed INTEGER,
    penalties_saved  INTEGER,
    ict_index        REAL,
    influence        REAL,
    creativity       REAL,
    threat           REAL,
    UNIQUE (player_code, season_id)
);

CREATE TABLE IF NOT EXISTS fact_gw_player (
    gw_player_sk        INTEGER PRIMARY KEY,
    season_id           INTEGER NOT NULL REFERENCES dim_season(season_id),
    gw                  INTEGER NOT NULL,
    fixture_id          INTEGER NOT NULL,
    player_code         INTEGER NOT NULL REFERENCES dim_player(player_code),
    fpl_id              INTEGER NOT NULL,
    team_sk             INTEGER REFERENCES dim_team(team_sk),
    opponent_team_sk    INTEGER REFERENCES dim_team(team_sk),
    position_code       INTEGER,
    position_label      TEXT,
    kickoff_time        TEXT,
    was_home            INTEGER,
    minutes             INTEGER NOT NULL DEFAULT 0,
    starts              INTEGER,
    total_points        INTEGER NOT NULL DEFAULT 0,
    goals_scored        INTEGER NOT NULL DEFAULT 0,
    assists             INTEGER NOT NULL DEFAULT 0,
    clean_sheets        INTEGER NOT NULL DEFAULT 0,
    goals_conceded      INTEGER NOT NULL DEFAULT 0,
    own_goals           INTEGER NOT NULL DEFAULT 0,
    penalties_missed    INTEGER NOT NULL DEFAULT 0,
    penalties_saved     INTEGER NOT NULL DEFAULT 0,
    yellow_cards        INTEGER NOT NULL DEFAULT 0,
    red_cards           INTEGER NOT NULL DEFAULT 0,
    saves               INTEGER NOT NULL DEFAULT 0,
    bonus               INTEGER NOT NULL DEFAULT 0,
    bps                 INTEGER NOT NULL DEFAULT 0,
    ict_index           REAL,
    influence           REAL,
    creativity          REAL,
    threat              REAL,
    value               INTEGER NOT NULL,
    selected            INTEGER,
    transfers_in        INTEGER,
    transfers_out       INTEGER,
    transfers_balance   INTEGER,
    team_h_score        INTEGER,
    team_a_score        INTEGER,
    xp                  REAL,
    expected_goals      REAL,
    expected_assists    REAL,
    expected_goal_involvements  REAL,
    expected_goals_conceded     REAL,
    modified            INTEGER,
    -- Old Opta era (2016-17 to 2018-19)
    big_chances_created             INTEGER,
    big_chances_missed              INTEGER,
    key_passes                      INTEGER,
    completed_passes                INTEGER,
    attempted_passes                INTEGER,
    dribbles                        INTEGER,
    fouls                           INTEGER,
    offside                         INTEGER,
    errors_leading_to_goal          INTEGER,
    errors_leading_to_goal_attempt  INTEGER,
    open_play_crosses               INTEGER,
    target_missed                   INTEGER,
    winning_goals                   INTEGER,
    -- Old Opta + 2025-26
    clearances_blocks_interceptions INTEGER,
    recoveries                      INTEGER,
    tackles                         INTEGER,
    -- 2025-26 only
    defensive_contribution          REAL,
    -- Manager era (2024-25 only)
    mng_win             INTEGER,
    mng_draw            INTEGER,
    mng_loss            INTEGER,
    mng_goals_scored    INTEGER,
    mng_clean_sheets    INTEGER,
    mng_underdog_win    INTEGER,
    mng_underdog_draw   INTEGER,
    UNIQUE (season_id, gw, fpl_id, fixture_id)
);

CREATE INDEX IF NOT EXISTS idx_fgp_player_ts  ON fact_gw_player (player_code, season_id, gw);
CREATE INDEX IF NOT EXISTS idx_fgp_team_gw    ON fact_gw_player (team_sk, season_id, gw);
CREATE INDEX IF NOT EXISTS idx_fgp_opponent   ON fact_gw_player (opponent_team_sk, season_id, gw);
CREATE INDEX IF NOT EXISTS idx_fgp_pos_season ON fact_gw_player (position_code, season_id);
CREATE INDEX IF NOT EXISTS idx_dps_fplid      ON dim_player_season (season_id, fpl_id);
CREATE INDEX IF NOT EXISTS idx_dt_season_tid  ON dim_team (season_id, team_id);
"""
