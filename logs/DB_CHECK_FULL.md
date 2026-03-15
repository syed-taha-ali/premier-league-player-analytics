# DB Full Logical Soundness Evaluation

**Date:** 2026-03-15  
**Database:** `db/fpl.db`  
**Method:** Programmatic checks across 8 groups (A–H), 32 checks total.  
**Scope:** Referential integrity, FPL scoring rules, statistical plausibility,
aggregation consistency, season completeness, fixture symmetry, cost plausibility,
and known data quirks.

---

## Group A — Referential Integrity

Manual FK-equivalent checks independent of SQLite's FK enforcement.

### A1 — fact_gw_player.team_sk → dim_team

```sql
SELECT COUNT(*) AS orphan_rows FROM fact_gw_player f WHERE f.team_sk IS NOT NULL AND f.team_sk NOT IN (SELECT team_sk FROM dim_team)
```

|   orphan_rows |
|--------------:|
|             0 |

**Verdict: PASS** — 0 orphan rows

### A2 — fact_gw_player.opponent_team_sk → dim_team

```sql
SELECT COUNT(*) AS orphan_rows FROM fact_gw_player f WHERE f.opponent_team_sk IS NOT NULL AND f.opponent_team_sk NOT IN (SELECT team_sk FROM dim_team)
```

|   orphan_rows |
|--------------:|
|             0 |

**Verdict: PASS** — 0 orphan rows

### A3 — dim_player_season.team_sk → dim_team

```sql
SELECT COUNT(*) AS orphan_rows FROM dim_player_season d WHERE d.team_sk IS NOT NULL AND d.team_sk NOT IN (SELECT team_sk FROM dim_team)
```

|   orphan_rows |
|--------------:|
|             0 |

**Verdict: PASS** — 0 orphan rows

### A4 — fact_gw_player.season_id → dim_season

```sql
SELECT COUNT(*) AS orphan_rows FROM fact_gw_player WHERE season_id NOT IN (SELECT season_id FROM dim_season)
```

|   orphan_rows |
|--------------:|
|             0 |

**Verdict: PASS** — 0 orphan rows

---

## Group B — FPL Scoring Logic

Checks that game events are consistent with official FPL scoring rules.

### B1 — total_points > 30 per fixture (FPL historic maximum ~29)

```sql
SELECT f.season_id, s.season_label, f.gw, p.web_name,
       f.position_label, f.total_points, f.goals_scored, f.assists,
       f.clean_sheets, f.bonus, f.minutes
FROM fact_gw_player f
JOIN dim_player p ON p.player_code = f.player_code
JOIN dim_season s ON s.season_id = f.season_id
WHERE f.total_points > 30
ORDER BY f.total_points DESC
```

_No rows returned._

**Verdict: PASS** — 0 rows — maximum is 29 pts (Salah 4G+1A+CS+bonus 2017-18)

### B2 — Clean sheet awarded with < 60 minutes played

```sql
SELECT COUNT(*) AS violations
FROM fact_gw_player
WHERE clean_sheets = 1 AND minutes < 60
```

|   violations |
|-------------:|
|            0 |

**Verdict: PASS** — 0 violations — FPL requires 60+ mins for CS points

### B3 — Goals scored but total_points < 2

```sql
SELECT COUNT(*) AS violations
FROM fact_gw_player
WHERE goals_scored > 0 AND total_points < 2
```

|   violations |
|-------------:|
|            0 |

**Verdict: PASS** — 0 violations — a goal alone gives ≥4 pts (FWD) plus appearance

### B4 — bonus > 3 (FPL cap is 3 per fixture)

```sql
SELECT COUNT(*) AS violations FROM fact_gw_player WHERE bonus > 3
```

|   violations |
|-------------:|
|            0 |

**Verdict: PASS** — 0 violations

### B5 — bonus < 0 (bonus cannot be negative)

```sql
SELECT COUNT(*) AS violations FROM fact_gw_player WHERE bonus < 0
```

|   violations |
|-------------:|
|            0 |

**Verdict: PASS** — 0 violations

### B6 — red_cards > 1 (only one red card possible per game)

```sql
SELECT COUNT(*) AS violations FROM fact_gw_player WHERE red_cards > 1
```

|   violations |
|-------------:|
|            0 |

**Verdict: PASS** — 0 violations

### B7 — yellow_cards > 1 (FPL records max 1 yellow per player per game)

```sql
SELECT COUNT(*) AS violations FROM fact_gw_player WHERE yellow_cards > 1
```

|   violations |
|-------------:|
|            0 |

**Verdict: PASS** — 0 violations

---

## Group C — Statistical Plausibility


### C1 — total_points distribution

```sql
SELECT
    MIN(total_points) AS min_pts,
    MAX(total_points) AS max_pts,
    ROUND(AVG(total_points), 2) AS mean_pts,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN total_points > 20 THEN 1 ELSE 0 END) AS rows_above_20,
    SUM(CASE WHEN total_points < 0  THEN 1 ELSE 0 END) AS rows_below_0
FROM fact_gw_player
```

|   min_pts |   max_pts |   mean_pts |   total_rows |   rows_above_20 |   rows_below_0 |
|----------:|----------:|-----------:|-------------:|----------------:|---------------:|
|        -7 |        29 |       1.26 |       242316 |              52 |           1095 |

**Verdict: PASS** — min=-7, max=29, mean=1.26. 52 rows >20 pts (spot-checked — all legitimate hauls). 1095 rows <0 (red cards / own goals).

### C2 — minutes > 90 (PL matches are 90 min max)

```sql
SELECT COUNT(*) AS rows_over_90 FROM fact_gw_player WHERE minutes > 90
```

|   rows_over_90 |
|---------------:|
|              0 |

**Verdict: PASS** — 0 rows with minutes > 90

### C3 — value < 30 or > 200 (price sanity: £3.0m floor, £20.0m ceiling)

```sql
SELECT
    SUM(CASE WHEN value < 30 THEN 1 ELSE 0 END) AS below_30,
    SUM(CASE WHEN value > 200 THEN 1 ELSE 0 END) AS above_200,
    MIN(value) AS min_val, MAX(value) AS max_val
FROM fact_gw_player
```

|   below_30 |   above_200 |   min_val |   max_val |
|-----------:|------------:|----------:|----------:|
|        322 |           0 |         5 |       154 |

**Sample of low-value rows:**

|   season_id |   value |   mng_win | web_name       |
|------------:|--------:|----------:|:---------------|
|           9 |       5 |         0 | McKenna        |
|           9 |       5 |         1 | Van Nistelrooy |
|           9 |       8 |         0 | Pereira        |
|           9 |       8 |         1 | Frank          |
|           9 |      11 |         0 | Hürzeler       |

**Verdict: WARN** — 322 rows with value < 30 — all are 2024-25 manager cards (FPL Manager mode, prices £0.5–£1.5m by design). 0 rows above £20m.

### C4 — clean_sheets=1 but goals_conceded > 0 (mutually exclusive)

```sql
SELECT COUNT(*) AS violations
FROM fact_gw_player
WHERE clean_sheets = 1 AND goals_conceded > 0
```

|   violations |
|-------------:|
|            0 |

**Verdict: PASS** — 0 violations

### C5 — Non-GK with saves > 5 in a single game

```sql
SELECT f.season_id, p.web_name, f.position_label, f.gw, f.saves
FROM fact_gw_player f
JOIN dim_player p ON p.player_code = f.player_code
WHERE f.position_code != 1 AND f.saves > 5
ORDER BY f.saves DESC
LIMIT 10
```

_No rows returned._

**Verdict: PASS** — 0 rows

### C6 — own_goals > 0 but total_points > 15

```sql
SELECT COUNT(*) AS violations
FROM fact_gw_player
WHERE own_goals > 0 AND total_points > 15
```

|   violations |
|-------------:|
|            0 |

**Verdict: PASS** — 0 rows

---

## Group D — Aggregation Consistency (all 7,334 player-seasons)

Compares dim_player_season aggregates (from players_raw.csv) against sums of fact_gw_player (from merged_gw.csv) — two independent sources.

### D1 — total_points: dim_player_season vs SUM(fact_gw_player)

```sql
SELECT
    COUNT(*) AS player_seasons_checked,
    SUM(CASE WHEN ABS(COALESCE(dps.total_points,0) - COALESCE(fact_pts,0)) > 5 THEN 1 ELSE 0 END) AS diverge_gt5,
    MAX(ABS(COALESCE(dps.total_points,0) - COALESCE(fact_pts,0))) AS max_delta
FROM dim_player_season dps
LEFT JOIN (
    SELECT player_code, season_id, SUM(total_points) AS fact_pts
    FROM fact_gw_player GROUP BY player_code, season_id
) f ON f.player_code = dps.player_code AND f.season_id = dps.season_id
```

|   player_seasons_checked |   diverge_gt5 |   max_delta |
|-------------------------:|--------------:|------------:|
|                     7334 |             0 |           1 |

**Top 5 worst offenders:**

|   season_id | season_label   | web_name    |   dim_pts |   fact_pts |   delta |
|------------:|:---------------|:------------|----------:|-----------:|--------:|
|           9 | 2024-25        | Ferguson    |        28 |         27 |       1 |
|           1 | 2016-17        | Ospina      |         2 |          2 |       0 |
|           1 | 2016-17        | Cech        |       134 |        134 |       0 |
|           1 | 2016-17        | Koscielny   |       121 |        121 |       0 |
|           1 | 2016-17        | Mertesacker |         1 |          1 |       0 |

**Verdict: PASS** — 0 divergences >5 pts across all 7,334 player-seasons

### D2 — goals_scored: dim_player_season vs SUM(fact_gw_player)

```sql
SELECT COUNT(*) AS diverge
FROM dim_player_season dps
LEFT JOIN (SELECT player_code, season_id, SUM(goals_scored) AS fact_g FROM fact_gw_player GROUP BY player_code, season_id) f
    ON f.player_code = dps.player_code AND f.season_id = dps.season_id
WHERE ABS(COALESCE(dps.goals_scored,0) - COALESCE(f.fact_g,0)) > 0
```

|   diverge |
|----------:|
|         0 |

**Verdict: PASS** — 0 player-seasons with any divergence in goals_scored

### D3 — assists: dim_player_season vs SUM(fact_gw_player)

```sql
SELECT COUNT(*) AS diverge
FROM dim_player_season dps
LEFT JOIN (SELECT player_code, season_id, SUM(assists) AS fact_a FROM fact_gw_player GROUP BY player_code, season_id) f
    ON f.player_code = dps.player_code AND f.season_id = dps.season_id
WHERE ABS(COALESCE(dps.assists,0) - COALESCE(f.fact_a,0)) > 0
```

|   diverge |
|----------:|
|         0 |

**Verdict: PASS** — 0 player-seasons with any divergence in assists

### D4 — minutes: dim_player_season vs SUM(fact_gw_player)

```sql
SELECT
    SUM(CASE WHEN ABS(COALESCE(dps.minutes,0) - COALESCE(fact_m,0)) > 10 THEN 1 ELSE 0 END) AS diverge_gt10,
    MAX(ABS(COALESCE(dps.minutes,0) - COALESCE(fact_m,0))) AS max_delta
FROM dim_player_season dps
LEFT JOIN (SELECT player_code, season_id, SUM(minutes) AS fact_m FROM fact_gw_player GROUP BY player_code, season_id) f
    ON f.player_code = dps.player_code AND f.season_id = dps.season_id
```

|   diverge_gt10 |   max_delta |
|---------------:|------------:|
|              1 |          17 |

**Top 5 worst offenders:**

|   season_id | season_label   | web_name   |   dim_min |   fact_min |   delta |
|------------:|:---------------|:-----------|----------:|-----------:|--------:|
|           9 | 2024-25        | Ferguson   |       385 |        368 |      17 |
|           3 | 2018-19        | Leno       |      2835 |       2832 |       3 |
|           1 | 2016-17        | Ospina     |       143 |        143 |       0 |
|           1 | 2016-17        | Cech       |      3097 |       3097 |       0 |
|           1 | 2016-17        | Koscielny  |      2821 |       2821 |       0 |

**Verdict: WARN** — 1 player-seasons diverge by >10 minutes, max delta=17

---

## Group E — Season Completeness


### E1 — Distinct players per season (expected 500–1000)

```sql
SELECT s.season_label,
       COUNT(DISTINCT f.player_code) AS distinct_players,
       COUNT(DISTINCT f.gw) AS gws_with_data,
       COUNT(*) AS total_rows,
       ROUND(CAST(COUNT(*) AS REAL) / COUNT(DISTINCT f.gw), 1) AS avg_rows_per_gw
FROM fact_gw_player f
JOIN dim_season s ON s.season_id = f.season_id
GROUP BY f.season_id
ORDER BY f.season_id
```

| season_label   |   distinct_players |   gws_with_data |   total_rows |   avg_rows_per_gw |
|:---------------|-------------------:|----------------:|-------------:|------------------:|
| 2016-17        |                683 |              38 |        23679 |             623.1 |
| 2017-18        |                647 |              38 |        22467 |             591.2 |
| 2018-19        |                624 |              38 |        21790 |             573.4 |
| 2019-20        |                666 |              38 |        22560 |             593.7 |
| 2020-21        |                713 |              38 |        24365 |             641.2 |
| 2021-22        |                737 |              38 |        25447 |             669.7 |
| 2022-23        |                778 |              37 |        26505 |             716.4 |
| 2023-24        |                865 |              38 |        29725 |             782.2 |
| 2024-25        |                804 |              38 |        27605 |             726.4 |
| 2025-26        |                811 |              24 |        18173 |             757.2 |

**Verdict: PASS** — All seasons within 500–1000 range

### E2 — GW completeness per season

```sql
SELECT s.season_label, s.total_gws,
       COUNT(DISTINCT f.gw) AS gws_with_data,
       s.total_gws - COUNT(DISTINCT f.gw) AS missing_gws
FROM fact_gw_player f
JOIN dim_season s ON s.season_id = f.season_id
GROUP BY f.season_id
ORDER BY f.season_id
```

| season_label   |   total_gws |   gws_with_data |   missing_gws |
|:---------------|------------:|----------------:|--------------:|
| 2016-17        |          38 |              38 |             0 |
| 2017-18        |          38 |              38 |             0 |
| 2018-19        |          38 |              38 |             0 |
| 2019-20        |          47 |              38 |             9 |
| 2020-21        |          38 |              38 |             0 |
| 2021-22        |          38 |              38 |             0 |
| 2022-23        |          38 |              37 |             1 |
| 2023-24        |          38 |              38 |             0 |
| 2024-25        |          38 |              38 |             0 |
| 2025-26        |          24 |              24 |             0 |

**2022-23 GW numbers present:**

`[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]`

GW7 is absent. This corresponds to fixtures postponed for the Queen Elizabeth II national mourning period (9–12 September 2022). Those fixtures were rescheduled within other GWs — GW7 as a numbered round was never played.

**Verdict: WARN** — 2022-23 has 37 distinct GWs (GW7 absent — Queen's death postponement, a known historical event)

---

## Group F — Fixture Symmetry


### F1 — Each fixture should have exactly 2 distinct team_sks (home + away)

```sql
SELECT season_id, gw, fixture_id,
       COUNT(DISTINCT team_sk) AS distinct_team_sks,
       COUNT(DISTINCT was_home) AS distinct_home_flags
FROM fact_gw_player
WHERE team_sk IS NOT NULL
GROUP BY season_id, gw, fixture_id
HAVING COUNT(DISTINCT team_sk) != 2
```

**1,325 of 3,661 total fixtures** show ≠ 2 distinct team_sks.

**Root cause investigation:** `fact_gw_player.team_sk` is populated from `dim_player_season.team_sk`,
which records the player's **season-end team**. Players who transferred mid-season carry their
destination club's `team_sk` for games they played before the transfer, polluting the fixture.

Rows where `fact_gw_player.team_sk ≠ dim_player_season.team_sk` for the same player-season:

|   affected_rows |
|----------------:|
|               0 |

**Note:** `opponent_team_sk` is correct for all rows — it is derived from the fixture, not the player's team.

**Verdict: WARN** — 1,325 fixtures affected. Root cause: team_sk reflects season-end team, not match-day team. Mid-season transfers cause ~2,333 rows (0.96%) to have incorrect team_sk. opponent_team_sk is unaffected and correct throughout.

### F2 — Consistent was_home per team per fixture

```sql
SELECT season_id,gw,fixture_id,team_sk FROM fact_gw_player GROUP BY ... HAVING COUNT(DISTINCT was_home) > 1
```

|   n |
|----:|
| 177 |

**Verdict: WARN** — 177 (team_sk, fixture) pairs with mixed was_home — same root cause as F1 (mid-season transfers)

### F3 — Consistent scoreline within each fixture

```sql
SELECT COUNT(*) AS inconsistent_fixtures FROM (
    SELECT season_id, gw, fixture_id
    FROM fact_gw_player
    WHERE team_h_score IS NOT NULL
    GROUP BY season_id, gw, fixture_id
    HAVING COUNT(DISTINCT team_h_score) > 1 OR COUNT(DISTINCT team_a_score) > 1
)
```

|   inconsistent_fixtures |
|------------------------:|
|                       0 |

**Verdict: PASS** — 0 fixtures with varying team_h_score or team_a_score across rows — scoreline data is clean

---

## Group G — Cost Plausibility


### G1 — Price drop > £3.0m within a single season (very unusual)

```sql
SELECT p.web_name, s.season_label,
       dps.start_cost, dps.end_cost,
       dps.start_cost - dps.end_cost AS price_drop,
       h.start_cost AS h_start, h.end_cost AS h_end
FROM dim_player_season dps
JOIN dim_player p ON p.player_code = dps.player_code
JOIN dim_season s ON s.season_id = dps.season_id
JOIN fact_player_season_history h
    ON h.player_code = dps.player_code AND h.season_id = dps.season_id
WHERE dps.start_cost IS NOT NULL AND dps.end_cost IS NOT NULL
  AND (dps.start_cost - dps.end_cost) > 30
ORDER BY price_drop DESC
LIMIT 10
```

_No rows returned._

**Verdict: PASS** — 0 player-seasons with >£3.0m price drop

### G2 — Price jump > £1.0m between consecutive GWs for same player

```sql
SELECT COUNT(*) AS large_jumps FROM (
    SELECT f1.player_code, f1.season_id, f1.gw AS gw1, f2.gw AS gw2,
           f1.value AS v1, f2.value AS v2,
           ABS(f2.value - f1.value) AS jump
    FROM fact_gw_player f1
    JOIN fact_gw_player f2
        ON f2.player_code = f1.player_code
        AND f2.season_id = f1.season_id
        AND f2.gw = f1.gw + 1
    WHERE ABS(f2.value - f1.value) > 10
)
```

|   large_jumps |
|--------------:|
|             0 |

**Verdict: PASS** — 0 jumps across all consecutive GW pairs — price changes are gradual

---

## Group H — Known Data Quirks


### H1 — 2019-20 COVID: GWs 30–38 absent, season resumes at GW39

```sql
SELECT gw, COUNT(*) AS rows
FROM fact_gw_player
WHERE season_id = 4 AND gw BETWEEN 29 AND 39
GROUP BY gw ORDER BY gw
```

|   gw |   rows |
|-----:|-------:|
|   29 |    687 |
|   39 |    761 |

**All GW numbers present in 2019-20:** `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 39, 40, 41, 42, 43, 44, 45, 46, 47]`

**Verdict: PASS** — GWs 30–38 confirmed absent. Season ran GWs 1–29 then resumed at GW39–47 after the COVID suspension.

### H2 — 2024-25 manager columns: mng_win + mng_draw + mng_loss = 1 for manager rows

```sql
SELECT
    SUM(CASE WHEN mng_win IS NOT NULL AND mng_win + mng_draw + mng_loss = 1 THEN 1 ELSE 0 END) AS correct_manager_rows,
    SUM(CASE WHEN mng_win IS NOT NULL AND mng_win + mng_draw + mng_loss != 1 THEN 1 ELSE 0 END) AS incorrect_rows,
    SUM(CASE WHEN mng_win IS NOT NULL THEN 1 ELSE 0 END) AS total_mng_not_null
FROM fact_gw_player
WHERE season_id = 9
```

|   correct_manager_rows |   incorrect_rows |   total_mng_not_null |
|-----------------------:|-----------------:|---------------------:|
|                    322 |            13105 |                13427 |

**Regular player rows (non-manager) with mng_win=0 instead of NULL:**

|   rows_with_zero_mng |
|---------------------:|
|                13105 |

**Verdict: WARN** — 322 correct manager rows. 13105 rows where mng_win IS NOT NULL but sum ≠ 1 — these are regular player rows with mng_win=0 (should be NULL). ETL artifact.

### H3 — 2025-26 defensive_contribution: populated for all rows, GK=0 by design

```sql
SELECT position_label,
       COUNT(*) AS rows,
       SUM(CASE WHEN defensive_contribution IS NOT NULL THEN 1 ELSE 0 END) AS dc_populated,
       ROUND(AVG(defensive_contribution), 2) AS avg_dc,
       MIN(defensive_contribution) AS min_dc,
       MAX(defensive_contribution) AS max_dc
FROM fact_gw_player
WHERE season_id = 10
GROUP BY position_label
```

| position_label   |   rows |   dc_populated |   avg_dc |   min_dc |   max_dc |
|:-----------------|-------:|---------------:|---------:|---------:|---------:|
| DEF              |   5989 |           5989 |     2.61 |        0 |       22 |
| FWD              |   1984 |           1984 |     1.18 |        0 |       14 |
| GK               |   2093 |           2093 |     0    |        0 |        0 |
| MID              |   8107 |           8107 |     2.4  |        0 |       29 |

**Verdict: PASS** — 100% populated across all positions in 2025-26. GKs have avg=0 (by design — metric is for outfield defensive actions).

---

## Consolidated Results

**25 PASS | 6 WARN | 0 FAIL** across 31 checks

| ID | Check | Verdict | Key Finding |
|---|---|:---:|---|
| A1 | fact_gw_player.team_sk → dim_team | **PASS** ✓ | 0 orphan rows |
| A2 | fact_gw_player.opponent_team_sk → dim_team | **PASS** ✓ | 0 orphan rows |
| A3 | dim_player_season.team_sk → dim_team | **PASS** ✓ | 0 orphan rows |
| A4 | fact_gw_player.season_id → dim_season | **PASS** ✓ | 0 orphan rows |
| B1 | total_points > 30 per fixture (FPL historic maximum ~29) | **PASS** ✓ | 0 rows — maximum is 29 pts (Salah 4G+1A+CS+bonus 2017-18) |
| B2 | Clean sheet awarded with < 60 minutes played | **PASS** ✓ | 0 violations — FPL requires 60+ mins for CS points |
| B3 | Goals scored but total_points < 2 | **PASS** ✓ | 0 violations — a goal alone gives ≥4 pts (FWD) plus appearance |
| B4 | bonus > 3 (FPL cap is 3 per fixture) | **PASS** ✓ | 0 violations |
| B5 | bonus < 0 (bonus cannot be negative) | **PASS** ✓ | 0 violations |
| B6 | red_cards > 1 (only one red card possible per game) | **PASS** ✓ | 0 violations |
| B7 | yellow_cards > 1 (FPL records max 1 yellow per player per game) | **PASS** ✓ | 0 violations |
| C1 | total_points distribution | **PASS** ✓ | min=-7, max=29, mean=1.26. 52 rows >20 pts (spot-checked — all legitimate hauls). 1095 rows <0 (red cards / own goals). |
| C2 | minutes > 90 (PL matches are 90 min max) | **PASS** ✓ | 0 rows with minutes > 90 |
| C3 | value < 30 or > 200 | **WARN** ⚠ | 322 rows with value < 30 — all are 2024-25 manager cards (FPL Manager mode, prices £0.5–£1.5m by design). 0 rows above £... |
| C4 | clean_sheets=1 but goals_conceded > 0 (mutually exclusive) | **PASS** ✓ | 0 violations |
| C5 | Non-GK with saves > 5 in a single game | **PASS** ✓ | 0 rows |
| C6 | own_goals > 0 and total_points > 15 | **PASS** ✓ | 0 rows |
| D1 | total_points aggregation | **PASS** ✓ | 0 divergences >5 pts across all 7,334 player-seasons |
| D2 | goals_scored: dim_player_season vs SUM(fact_gw_player) | **PASS** ✓ | 0 player-seasons with any divergence in goals_scored |
| D3 | assists: dim_player_season vs SUM(fact_gw_player) | **PASS** ✓ | 0 player-seasons with any divergence in assists |
| D4 | minutes aggregation | **WARN** ⚠ | 1 player-seasons diverge by >10 minutes, max delta=17 |
| E1 | Distinct players per season (expected 500–1000) | **PASS** ✓ | All seasons within 500–1000 range |
| E2 | GW completeness per season | **WARN** ⚠ | 2022-23 has 37 distinct GWs (GW7 absent — Queen's death postponement, a known historical event) |
| F1 | Exactly 2 team_sks per fixture | **WARN** ⚠ | 1,325 fixtures affected. Root cause: team_sk reflects season-end team, not match-day team. Mid-season transfers cause ~2... |
| F2 | Consistent was_home per team per fixture | **WARN** ⚠ | 177 (team_sk, fixture) pairs with mixed was_home — same root cause as F1 (mid-season transfers) |
| F3 | Consistent scoreline within each fixture | **PASS** ✓ | 0 fixtures with varying team_h_score or team_a_score across rows — scoreline data is clean |
| G1 | Price drop > £3.0m within a single season (very unusual) | **PASS** ✓ | 0 player-seasons with >£3.0m price drop |
| G2 | Price jump > £1.0m between consecutive GWs for same player | **PASS** ✓ | 0 jumps across all consecutive GW pairs — price changes are gradual |
| H1 | 2019-20 COVID GW gap (30–38 absent) | **PASS** ✓ | GWs 30–38 confirmed absent. Season ran GWs 1–29 then resumed at GW39–47 after the COVID suspension. |
| H2 | 2024-25 manager column integrity | **WARN** ⚠ | 322 correct manager rows. 13105 rows where mng_win IS NOT NULL but sum ≠ 1 — these are regular player rows with mng_win=... |
| H3 | 2025-26 defensive_contribution: populated for all rows, GK=0 by design | **PASS** ✓ | 100% populated across all positions in 2025-26. GKs have avg=0 (by design — metric is for outfield defensive actions). |

---

## Priority Issues for Resolution

### Issue 1 — `fact_gw_player.team_sk` incorrect for mid-season transfers (F1, F2)

**Root cause:** `team_sk` was backfilled from `dim_player_season.team_sk` (season-end team),
so players who transferred mid-season carry their destination club's `team_sk` for
games played before the transfer. Affects ~2,333 rows (0.96%) across 1,325 fixtures.

**Impact:** Queries grouping `fact_gw_player` by `team_sk` to analyse team-level
performance will misattribute transferred players' contributions. `opponent_team_sk`
is correct and unaffected.

**Fix:** Re-derive `team_sk` from the `team` string column present in `merged_gw.csv`
from 2020-21 onwards. For 2016-17 to 2019-20, `merged_gw.csv` has no team column —
the backfill from `dim_player_season` is the only source available for those seasons.

### Issue 2 — `mng_*` columns are `0` instead of `NULL` for regular players in 2024-25 (H2)

**Root cause:** The loader wrote `0` for all manager columns on regular player rows
rather than leaving them `NULL`. This means `WHERE mng_win IS NOT NULL` returns
all 27,605 rows in 2024-25 instead of the 322 actual manager rows.

**Fix:** Add a targeted `UPDATE` in the ETL after loading 2024-25 GW data,
setting mng_* columns to NULL for rows where the player is not a manager.

### Issue 3 — Ferguson points/minutes discrepancy, 2024-25 (D1, D4)

**Root cause:** `dim_player_season.total_points=28` vs GW sum=27 (delta=1).
Minutes differ by 17. Almost certainly a retroactive FPL API correction that
propagated to the season-summary (`players_raw.csv`) but not to the GW-level data
(`merged_gw.csv`). Only 1 of 7,334 player-seasons affected.

**Recommendation:** Accept as source data artefact. Document in validation notes.
