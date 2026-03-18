# FPL Star Schema Design Plan

## Context

We are designing the authoritative relational schema that the FPL data science pipeline will be built on. The raw dataset spans 10 seasons (2016-17 to 2025-26), ~13,923 CSV files, with significant schema drift across seasons. The goal is a clean star schema suitable for analytics and ML modelling that uses the minimal set of authoritative source files, avoids reprocessing redundant data, and handles all known cross-season pitfalls (identifier reuse, column drift, missing team/position for early seasons, outdated aggregates).

---

## 1. Authoritative Sources (Load These)

| # | File Pattern | Count | Tables Fed |
|---|---|---|---|
| 1 | `{season}/gws/merged_gw.csv` | 10 | `fact_gw_player` (primary) |
| 2 | `{season}/players_raw.csv` | 10 | `dim_player`, `dim_player_season`, `dim_team` (2024-25/2025-26) |
| 3 | `data/master_team_list.csv` | 1 | `dim_team` (seasons 2016-17 to 2023-24) |
| 4 | `{season}/players/{name}/history.csv` | ~6,000 | `fact_player_season_history` (supplementary) |

## 2. Excluded Files (Reasons)

| File Pattern | Reason |
|---|---|
| `{season}/gws/gw{N}.csv` | Superseded by `merged_gw.csv` — same data split by GW |
| `{season}/gws/xP{N}.csv` | `xP` column already present in `merged_gw.csv` from 2020-21 |
| `{season}/cleaned_players.csv` | Strict subset of `players_raw.csv` with fewer columns |
| `{season}/players/{name}/gw.csv` | Identical to `merged_gw.csv` data, split by player — redundant |
| `{season}/player_idlist.csv` | Fully derivable from `players_raw.csv` (`first_name`, `second_name`, `id`) |
| `data/cleaned_merged_seasons.csv` | Outdated — missing 2024-25 and 2025-26 seasons |
| `data/cleaned_merged_seasons_team_aggregated.csv` | Outdated — same coverage gap |

---

## 3. Critical Data Engineering Facts

### 3a. The Player Identity Problem
**`fpl_id`** (`players_raw.id` = `merged_gw.element`) is **season-scoped and resets each year**.
**`player_code`** (`players_raw.code`) is **stable across all seasons** — this is the canonical player key.

Every cross-season join on player identity MUST follow this bridge:
```
merged_gw.element (season-scoped)
  → JOIN players_raw.id (same season)
  → players_raw.code = player_code (stable)
```
Never join `merged_gw.element` directly across seasons.

### 3b. Team ID Instability
`opponent_team` in `merged_gw.csv` is a season-scoped numeric team_id (1–20). Team IDs are NOT stable across seasons (promoted/relegated clubs reuse them). The `dim_team` table must be keyed by a surrogate `(season_id, team_id)`.

### 3c. Missing Position & Team for 2016-17 to 2019-20
`merged_gw.csv` for these seasons has **no `position` or `team` (name) columns**. Derive from `dim_player_season` (which reads them from `players_raw.csv`).

### 3d. Team Name Resolution Gap for 2024-25 and 2025-26
`master_team_list.csv` covers only 2016-17 to 2023-24. For 2024-25 and 2025-26, build team_id → team_name mapping by joining:
- `players_raw.id` → `players_raw.team` (gives team_id INT)
- `merged_gw.element` → `merged_gw.team` (gives team name STRING, present from 2020-21)

Group by `players_raw.team` within the season to get the canonical name for each team_id.

### 3e. Merged GW Schema Eras (column availability by era)
| Era | Seasons | Key differences |
|---|---|---|
| **Old Opta** | 2016-17, 2017-18, 2018-19 | 55–56 cols. No `position`, no `team` (name), no `xP`, no xG. Has: `attempted_passes`, `big_chances_*`, `dribbles`, `key_passes`, `fouls`, `offside`, `ea_index`, etc. |
| **Stripped** | 2019-20 | 33 cols. No `position`, no `team`, no `xP`, no xG. GW1–GW47 (COVID). |
| **Modern core** | 2020-21, 2021-22 | 35 cols. Adds `position`, `team` (string), `xP`. No xG. No `starts`. |
| **xG era** | 2022-23, 2023-24 | 40 cols. Adds `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`, `starts`. |
| **Manager era** | 2024-25 | 49 cols. Adds `mng_win`, `mng_draw`, `mng_loss`, `mng_goals_scored`, `mng_clean_sheets`, `mng_underdog_win`, `mng_underdog_draw`, `modified`. |
| **Defensive era** | 2025-26 | 44 cols. Drops `mng_*`. Adds `clearances_blocks_interceptions`, `defensive_contribution`, `recoveries`, `tackles`. Drops `starts`. |

### 3f. history.csv Deduplication
~6,000 history files each contain a player's full prior history, heavily overlapping. Deduplicate on `(element_code, season_name)` — keep the row from the **latest source season's file** for each unique pair. The unique value of history.csv is `start_cost` and `end_cost`, not available from any other source.

---

## 4. Star Schema

### Loading Order (dependency sequence)
```
1. dim_season
2. dim_player
3. dim_team
4. dim_player_season
5. fact_player_season_history   (supplementary; cross-validate start_cost with dim_player_season)
6. fact_gw_player
```

---

### `dim_season`
**Source:** Derived from season directory names.

| Column | Type | Constraint | Notes |
|---|---|---|---|
| `season_id` | SMALLINT | PK | Synthetic: 1=2016-17 … 10=2025-26 |
| `season_label` | VARCHAR(7) | UNIQUE NOT NULL | e.g. `"2016-17"` |
| `start_year` | SMALLINT | NOT NULL | e.g. 2016 |
| `end_year` | SMALLINT | NOT NULL | e.g. 2017 |
| `total_gws` | SMALLINT | NOT NULL | 38 for all seasons except 2019-20 (47) |
| `has_position_in_gw` | BOOLEAN | NOT NULL | FALSE for 2016-17/2017-18/2018-19/2019-20; TRUE from 2020-21 |
| `has_xp` | BOOLEAN | NOT NULL | FALSE pre-2020-21; TRUE from 2020-21 |
| `has_xg_stats` | BOOLEAN | NOT NULL | FALSE pre-2022-23; TRUE from 2022-23 |
| `has_starts` | BOOLEAN | NOT NULL | FALSE pre-2022-23 and for 2025-26; TRUE for 2022-23 and 2023-24 and 2024-25 |
| `has_mng_cols` | BOOLEAN | NOT NULL | TRUE for 2024-25 only |
| `team_map_source` | VARCHAR(20) | NOT NULL | `'master_team_list'` (seasons 1–8); `'derived'` (seasons 9–10) |

---

### `dim_player`
**Source:** Union of all 10 `players_raw.csv` files on `code` column. Latest season's values win on conflict.

| Column | Type | Constraint | Notes |
|---|---|---|---|
| `player_code` | INTEGER | PK | `players_raw.code` — stable cross-season |
| `first_name` | VARCHAR(100) | NOT NULL | From most-recent season's `players_raw.first_name` |
| `second_name` | VARCHAR(100) | NOT NULL | From most-recent season's `players_raw.second_name` |
| `web_name` | VARCHAR(100) | NOT NULL | From most-recent season's `players_raw.web_name` |
| `birth_date` | DATE | NULLABLE | Not in 2016-17; take first non-null across seasons |
| `region` | INTEGER | NULLABLE | Not in 2016-17; take first non-null across seasons |
| `debut_season_id` | SMALLINT | FK → dim_season | MIN(season_id) where player_code appears |

---

### `dim_team`
**Source:** `master_team_list.csv` (seasons 1–8) + derived for seasons 9–10 (see §3d).

| Column | Type | Constraint | Notes |
|---|---|---|---|
| `team_sk` | INTEGER | PK | Synthetic surrogate |
| `season_id` | SMALLINT | FK → dim_season NOT NULL | |
| `team_id` | SMALLINT | NOT NULL | Season-scoped FPL team ID (1–20) |
| `team_name` | VARCHAR(100) | NOT NULL | Full team name string |
| `team_code` | INTEGER | NULLABLE | Stable cross-season internal code (from `players_raw.team_code`); NULL when not derivable |

**Unique constraint:** `(season_id, team_id)`

---

### `dim_player_season`
**Source:** `{season}/players_raw.csv` (one row per player per season). Supplemented by `history.csv` for `start_cost`/`end_cost`.

| Column | Type | Constraint | Notes |
|---|---|---|---|
| `season_id` | SMALLINT | PK (composite), FK → dim_season | |
| `player_code` | INTEGER | PK (composite), FK → dim_player | |
| `fpl_id` | SMALLINT | NOT NULL | `players_raw.id` — season-scoped. Used for within-season joins only. |
| `team_sk` | INTEGER | FK → dim_team NULLABLE | `players_raw.team` (team_id) → dim_team lookup on (season_id, team_id) |
| `position_code` | SMALLINT | NOT NULL | `players_raw.element_type`. Map 5 → 3 (AM → MID) |
| `position_label` | VARCHAR(3) | NOT NULL | Derived: 1→GK, 2→DEF, 3→MID, 4→FWD |
| `start_cost` | SMALLINT | NULLABLE | `players_raw.now_cost - players_raw.cost_change_start`. Override with `history.csv.start_cost` where available. In £0.1m. |
| `end_cost` | SMALLINT | NULLABLE | `history.csv.end_cost`. NULL when history.csv absent. Do NOT use `now_cost` as end_cost proxy. |
| `status` | VARCHAR(1) | NULLABLE | `players_raw.status` at time of data capture (a/d/i/s/u/n) |
| `selected_by_percent` | NUMERIC(5,2) | NULLABLE | `players_raw.selected_by_percent` |
| `total_points` | SMALLINT | NULLABLE | `players_raw.total_points` |
| `minutes` | INTEGER | NULLABLE | `players_raw.minutes` |
| `goals_scored` | SMALLINT | NULLABLE | `players_raw.goals_scored` |
| `assists` | SMALLINT | NULLABLE | `players_raw.assists` |
| `clean_sheets` | SMALLINT | NULLABLE | `players_raw.clean_sheets` |
| `saves` | SMALLINT | NULLABLE | `players_raw.saves` |
| `bonus` | SMALLINT | NULLABLE | `players_raw.bonus` |
| `bps` | INTEGER | NULLABLE | `players_raw.bps` |
| `yellow_cards` | SMALLINT | NULLABLE | `players_raw.yellow_cards` |
| `red_cards` | SMALLINT | NULLABLE | `players_raw.red_cards` |
| `transfers_in` | INTEGER | NULLABLE | `players_raw.transfers_in` |
| `transfers_out` | INTEGER | NULLABLE | `players_raw.transfers_out` |
| `ict_index` | NUMERIC(6,1) | NULLABLE | `players_raw.ict_index` |
| `influence` | NUMERIC(8,1) | NULLABLE | `players_raw.influence` |
| `creativity` | NUMERIC(8,1) | NULLABLE | `players_raw.creativity` |
| `threat` | NUMERIC(8,1) | NULLABLE | `players_raw.threat` |
| `season_xg` | NUMERIC(6,2) | NULLABLE | `players_raw.expected_goals`. Present from 2022-23; in players_raw from 2024-25 only. |
| `season_xa` | NUMERIC(6,2) | NULLABLE | `players_raw.expected_assists`. Same availability. |
| `season_xgi` | NUMERIC(6,2) | NULLABLE | `players_raw.expected_goal_involvements`. |
| `season_xgc` | NUMERIC(6,2) | NULLABLE | `players_raw.expected_goals_conceded`. |

**Unique constraint:** `(season_id, fpl_id)` — enforces fpl_id uniqueness within season.

---

### `fact_gw_player` ← PRIMARY FACT TABLE
**Source:** `{season}/gws/merged_gw.csv` × 10 files.
**Grain:** One row per player per gameweek. ~228,000 rows (10 seasons × ~600 players × ~38 GWs avg).

| Column | Type | Constraint | Notes |
|---|---|---|---|
| `gw_player_sk` | BIGINT | PK | Synthetic surrogate |
| `season_id` | SMALLINT | FK → dim_season NOT NULL | |
| `gw` | SMALLINT | NOT NULL | `merged_gw.GW`. 1–38 (1–47 for 2019-20) |
| `player_code` | INTEGER | FK → dim_player NOT NULL | Resolved: `element` + season → `players_raw.code` |
| `fpl_id` | SMALLINT | NOT NULL | `merged_gw.element`. Season-scoped reference only. |
| `team_sk` | INTEGER | FK → dim_team NULLABLE | Player's own team this GW. From `merged_gw.team` (2020-21+) or `dim_player_season.team_sk` (pre-2020-21). |
| `opponent_team_sk` | INTEGER | FK → dim_team NULLABLE | `merged_gw.opponent_team` (team_id) → dim_team via (season_id, team_id) |
| `position_code` | SMALLINT | NULLABLE | From `merged_gw.position` (2020-21+) or `dim_player_season.position_code` (pre-2020-21) |
| `position_label` | VARCHAR(3) | NULLABLE | Derived: GK/DEF/MID/FWD |
| `kickoff_time` | TIMESTAMP | NULLABLE | `merged_gw.kickoff_time` (ISO 8601 → TIMESTAMP) |
| `was_home` | BOOLEAN | NOT NULL | `merged_gw.was_home` |
| `minutes` | SMALLINT | NOT NULL | `merged_gw.minutes` |
| `starts` | SMALLINT | NULLABLE | `merged_gw.starts`. **NULL pre-2022-23 and in 2025-26** |
| `total_points` | SMALLINT | NOT NULL | `merged_gw.total_points` |
| `goals_scored` | SMALLINT | NOT NULL | |
| `assists` | SMALLINT | NOT NULL | |
| `clean_sheets` | SMALLINT | NOT NULL | |
| `goals_conceded` | SMALLINT | NOT NULL | |
| `own_goals` | SMALLINT | NOT NULL | |
| `penalties_missed` | SMALLINT | NOT NULL | |
| `penalties_saved` | SMALLINT | NOT NULL | |
| `yellow_cards` | SMALLINT | NOT NULL | |
| `red_cards` | SMALLINT | NOT NULL | |
| `saves` | SMALLINT | NOT NULL | |
| `bonus` | SMALLINT | NOT NULL | |
| `bps` | SMALLINT | NOT NULL | |
| `ict_index` | NUMERIC(5,1) | NULLABLE | |
| `influence` | NUMERIC(6,1) | NULLABLE | |
| `creativity` | NUMERIC(6,1) | NULLABLE | |
| `threat` | NUMERIC(6,1) | NULLABLE | |
| `value` | SMALLINT | NOT NULL | Player price at this GW in £0.1m |
| `selected` | INTEGER | NULLABLE | Number of FPL managers owning player at kickoff |
| `transfers_in` | INTEGER | NULLABLE | Transfers in this GW |
| `transfers_out` | INTEGER | NULLABLE | Transfers out this GW |
| `transfers_balance` | INTEGER | NULLABLE | `transfers_in - transfers_out` |
| `team_h_score` | SMALLINT | NULLABLE | |
| `team_a_score` | SMALLINT | NULLABLE | |
| `xp` | NUMERIC(5,2) | NULLABLE | `merged_gw.xP`. **NULL pre-2020-21** |
| `expected_goals` | NUMERIC(5,2) | NULLABLE | **NULL pre-2022-23** |
| `expected_assists` | NUMERIC(5,2) | NULLABLE | **NULL pre-2022-23** |
| `expected_goal_involvements` | NUMERIC(5,2) | NULLABLE | **NULL pre-2022-23** |
| `expected_goals_conceded` | NUMERIC(5,2) | NULLABLE | **NULL pre-2022-23** |
| `modified` | BOOLEAN | NULLABLE | `merged_gw.modified`. **NULL pre-2024-25** |
| — *Era-specific (nullable outside their era)* — | | | |
| `big_chances_created` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `big_chances_missed` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `key_passes` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `completed_passes` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `attempted_passes` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `dribbles` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `fouls` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `offside` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `errors_leading_to_goal` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `errors_leading_to_goal_attempt` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `open_play_crosses` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `target_missed` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `winning_goals` | SMALLINT | NULLABLE | 2016-17 to 2018-19 only |
| `clearances_blocks_interceptions` | SMALLINT | NULLABLE | 2016-17 to 2018-19 and 2025-26 |
| `recoveries` | SMALLINT | NULLABLE | 2016-17 to 2018-19 and 2025-26 |
| `tackles` | SMALLINT | NULLABLE | 2016-17 to 2018-19 and 2025-26 |
| `defensive_contribution` | NUMERIC(5,2) | NULLABLE | 2025-26 only |
| `mng_win` | SMALLINT | NULLABLE | 2024-25 only |
| `mng_draw` | SMALLINT | NULLABLE | 2024-25 only |
| `mng_loss` | SMALLINT | NULLABLE | 2024-25 only |
| `mng_goals_scored` | SMALLINT | NULLABLE | 2024-25 only |
| `mng_clean_sheets` | SMALLINT | NULLABLE | 2024-25 only |
| `mng_underdog_win` | SMALLINT | NULLABLE | 2024-25 only |
| `mng_underdog_draw` | SMALLINT | NULLABLE | 2024-25 only |

**Unique constraint:** `(season_id, gw, fpl_id)`
**Alternate natural key:** `(season_id, gw, player_code)`

---

### `fact_player_season_history` (Supplementary)
**Source:** `{season}/players/{name}/history.csv` (~6,000 files).
**Grain:** One row per player per described prior season.
**Primary unique value:** `start_cost` and `end_cost` per player per season — not available from any other source.

**Deduplication:** Collect all files across all 10 source seasons. For each `(element_code, season_name)` pair, keep the row from the **latest** source season file. Season_name format is `"YYYY/YY"` (e.g. `"2021/22"`).

| Column | Type | Constraint | Notes |
|---|---|---|---|
| `history_sk` | BIGINT | PK | Synthetic surrogate |
| `player_code` | INTEGER | FK → dim_player NOT NULL | `history.csv.element_code` |
| `season_id` | SMALLINT | FK → dim_season NOT NULL | Mapped from `history.csv.season_name` → season_id |
| `start_cost` | SMALLINT | NULLABLE | Season opening price in £0.1m. Authoritative. |
| `end_cost` | SMALLINT | NULLABLE | Season closing price in £0.1m. Authoritative. |
| `total_points` | SMALLINT | NULLABLE | |
| `minutes` | INTEGER | NULLABLE | |
| `goals_scored` | SMALLINT | NULLABLE | |
| `assists` | SMALLINT | NULLABLE | |
| `clean_sheets` | SMALLINT | NULLABLE | |
| `saves` | SMALLINT | NULLABLE | |
| `bonus` | SMALLINT | NULLABLE | |
| `bps` | INTEGER | NULLABLE | |
| `yellow_cards` | SMALLINT | NULLABLE | |
| `red_cards` | SMALLINT | NULLABLE | |
| `own_goals` | SMALLINT | NULLABLE | |
| `penalties_missed` | SMALLINT | NULLABLE | |
| `penalties_saved` | SMALLINT | NULLABLE | |
| `ict_index` | NUMERIC(6,1) | NULLABLE | |
| `influence` | NUMERIC(8,1) | NULLABLE | |
| `creativity` | NUMERIC(8,1) | NULLABLE | |
| `threat` | NUMERIC(8,1) | NULLABLE | |

**Unique constraint:** `(player_code, season_id)`

---

## 5. FK Relationship Diagram

```
dim_season ──────────────────────────────────────────────────────────────┐
  │ season_id PK                                                          │
  ├──→ dim_team.season_id                                                 │
  ├──→ dim_player_season.season_id                                        │
  ├──→ fact_gw_player.season_id                                           │
  └──→ fact_player_season_history.season_id                               │
                                                                          │
dim_player ──────────────────────────────────────────────────────────────┤
  │ player_code PK                                                        │
  ├──→ dim_player_season.player_code                                      │
  ├──→ fact_gw_player.player_code                                         │
  ├──→ fact_player_season_history.player_code                             │
  └──→ dim_player.debut_season_id → dim_season.season_id ─────────────────┘

dim_team
  │ team_sk PK  (season_id FK, team_id)
  ├──→ dim_player_season.team_sk
  ├──→ fact_gw_player.team_sk           (player's own team)
  └──→ fact_gw_player.opponent_team_sk  (opponent)

dim_player_season
  │ (season_id, player_code) composite PK
  └── logically referenced by fact_gw_player for position/team backfill
      when merged_gw.csv lacks those columns (2016-17 to 2019-20)
```

---

## 6. Key Recommended Indexes

```sql
-- fact_gw_player
CREATE UNIQUE INDEX uix_fgp_grain       ON fact_gw_player (season_id, gw, fpl_id);
CREATE INDEX idx_fgp_player_ts          ON fact_gw_player (player_code, season_id, gw);
CREATE INDEX idx_fgp_team_gw            ON fact_gw_player (team_sk, season_id, gw);
CREATE INDEX idx_fgp_opponent           ON fact_gw_player (opponent_team_sk, season_id, gw);
CREATE INDEX idx_fgp_position_season    ON fact_gw_player (position_code, season_id);

-- dim_player_season
CREATE UNIQUE INDEX uix_dps_fplid       ON dim_player_season (season_id, fpl_id);

-- dim_team
CREATE UNIQUE INDEX uix_dt_season_tid   ON dim_team (season_id, team_id);
```

---

## 7. Verification Plan

1. **Row count sanity check:** `fact_gw_player` should have ~220,000–240,000 rows. Any season with 0 rows = load failure.
2. **No orphan FKs:** All `player_code` in fact tables must exist in `dim_player`. All `team_sk` must exist in `dim_team`. Run FK assertions after each table load.
3. **player_code bridge validation:** For each season, assert `COUNT(DISTINCT merged_gw.element) = COUNT(DISTINCT players_raw.id JOIN on same season)`. Any element with no matching player_code = data quarantine.
4. **dim_team completeness:** Assert exactly 20 team_id values per season_id in `dim_team`.
5. **Season-level point totals:** For each player-season, assert `SUM(fact_gw_player.total_points) ≈ dim_player_season.total_points` (within ±1 for rounding). Flag divergences > 5 points.
6. **start_cost cross-validation:** For each (player_code, season_id) where both `dim_player_season.start_cost` (derived) and `fact_player_season_history.start_cost` (history.csv) exist, assert they agree within ±1.
7. **NULL profile audit:** For each column with known NULL eras (xp, expected_*, starts), confirm NULLs appear in exactly the expected seasons and nowhere else.
8. **GW range check:** For each season, assert `MIN(gw) = 1` and `MAX(gw) = dim_season.total_gws`.
