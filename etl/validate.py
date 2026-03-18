"""
Post-load validation checks (docs/schema_design.md §7).
Prints PASS / FAIL for each check. Raises on any failure.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from .schema import SEASONS


def _q(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Execute sql on conn and return the scalar result."""
    return pd.read_sql(sql, conn)


def _check(name: str, passed: bool, detail: str = '') -> None:
    """Assert result == expected, printing PASS or FAIL; raise SystemExit on failure."""
    status = 'PASS' if passed else 'FAIL'
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not passed:
        raise AssertionError(msg)


def run_all(conn: sqlite3.Connection) -> None:
    """Run all 10 post-load validation checks; raises AssertionError if any fail."""
    print("\nRunning validation checks...")

    # 1. Row count sanity
    counts = _q(conn, "SELECT season_id, COUNT(*) AS n FROM fact_gw_player GROUP BY season_id")
    empty = counts[counts['n'] == 0]
    _check("Row count per season (no empty seasons)",
           empty.empty,
           f"Empty: {empty['season_id'].tolist()}" if not empty.empty else "")

    total = counts['n'].sum()
    _check("fact_gw_player total rows in expected range",
           200_000 <= total <= 280_000,  # expected row count bracket for 10 seasons; widen when season 11 data is added
           f"{total:,} rows")

    # 2. Orphan player_code in fact_gw_player
    orphans = _q(conn, """
        SELECT COUNT(*) AS n FROM fact_gw_player f
        WHERE NOT EXISTS (SELECT 1 FROM dim_player d WHERE d.player_code = f.player_code)
    """)['n'].iloc[0]
    _check("No orphan player_code in fact_gw_player", orphans == 0, f"{orphans} orphans")

    # 3. Orphan player_code in fact_player_season_history
    orphans_h = _q(conn, """
        SELECT COUNT(*) AS n FROM fact_player_season_history h
        WHERE NOT EXISTS (SELECT 1 FROM dim_player d WHERE d.player_code = h.player_code)
    """)['n'].iloc[0]
    _check("No orphan player_code in fact_player_season_history", orphans_h == 0, f"{orphans_h} orphans")

    # 4. player_code bridge: every element in merged_gw resolved to a player_code
    # (proxy: no NULLs on player_code in fact_gw_player)
    null_codes = _q(conn, "SELECT COUNT(*) AS n FROM fact_gw_player WHERE player_code IS NULL")['n'].iloc[0]
    _check("No NULL player_code in fact_gw_player", null_codes == 0, f"{null_codes} NULLs")

    # 5. dim_team completeness: 20 teams per season
    team_counts = _q(conn, "SELECT season_id, COUNT(*) AS n FROM dim_team GROUP BY season_id")
    wrong = team_counts[team_counts['n'] != 20]
    _check("Exactly 20 teams per season in dim_team",
           wrong.empty,
           f"Seasons with wrong count: {wrong.to_dict('records')}" if not wrong.empty else "")

    # 6. Season-level points reconciliation
    # SUM(fact_gw_player.total_points) per (player_code, season_id) ≈ dim_player_season.total_points
    # Excludes the current in-progress season: retroactive GW-level corrections by FPL accumulate
    # in the season total (players_raw.csv) but are not back-propagated to individual GW rows,
    # so the in-progress season will always show divergence beyond ±5 for corrected players.
    current_season = _q(conn, "SELECT MAX(season_id) AS s FROM fact_gw_player")['s'].iloc[0]
    reconcile = _q(conn, f"""
        SELECT
            dps.season_id,
            dps.player_code,
            dps.total_points AS dim_pts,
            COALESCE(SUM(f.total_points), 0) AS fact_pts
        FROM dim_player_season dps
        LEFT JOIN fact_gw_player f
            ON f.player_code = dps.player_code AND f.season_id = dps.season_id
        WHERE dps.season_id < {current_season}
        GROUP BY dps.season_id, dps.player_code
        HAVING ABS(COALESCE(SUM(f.total_points), 0) - COALESCE(dps.total_points, 0)) > 5  -- ±5 pt tolerance for retroactive FPL score corrections
    """)
    _check("Season point totals reconcile (within ±5, completed seasons only)",
           reconcile.empty,
           f"{len(reconcile)} player-seasons diverge" if not reconcile.empty else "")

    # 7. start_cost cross-validation (dim_player_season vs fact_player_season_history)
    cost_mismatch = _q(conn, """
        SELECT COUNT(*) AS n
        FROM dim_player_season dps
        JOIN fact_player_season_history h
            ON h.player_code = dps.player_code AND h.season_id = dps.season_id
        WHERE dps.start_cost IS NOT NULL
          AND h.start_cost IS NOT NULL
          AND ABS(dps.start_cost - h.start_cost) > 1
    """)['n'].iloc[0]
    _check("start_cost agrees within ±1 between dim_player_season and history",
           cost_mismatch == 0,
           f"{cost_mismatch} mismatches")

    # 8. GW range per season
    gw_ranges = _q(conn, """
        SELECT f.season_id, MIN(f.gw) AS min_gw, MAX(f.gw) AS max_gw, s.total_gws
        FROM fact_gw_player f
        JOIN dim_season s ON s.season_id = f.season_id
        GROUP BY f.season_id
    """)
    bad_range = gw_ranges[(gw_ranges['min_gw'] != 1) | (gw_ranges['max_gw'] != gw_ranges['total_gws'])]
    _check("GW range correct per season (min=1, max=total_gws)",
           bad_range.empty,
           f"Bad seasons: {bad_range[['season_id', 'min_gw', 'max_gw', 'total_gws']].to_dict('records')}" if not bad_range.empty else "")

    # 9. NULL profile: xp must be NULL pre-2020-21 (season_id < 5)
    bad_xp = _q(conn, """
        SELECT COUNT(*) AS n FROM fact_gw_player
        WHERE season_id < 5 AND xp IS NOT NULL
    """)['n'].iloc[0]
    _check("xp is NULL for pre-2020-21 seasons", bad_xp == 0, f"{bad_xp} unexpected non-NULLs")

    # 10. NULL profile: expected_goals must be NULL pre-2022-23 (season_id < 7)
    bad_xg = _q(conn, """
        SELECT COUNT(*) AS n FROM fact_gw_player
        WHERE season_id < 7 AND expected_goals IS NOT NULL
    """)['n'].iloc[0]
    _check("expected_goals is NULL for pre-2022-23 seasons", bad_xg == 0, f"{bad_xg} unexpected non-NULLs")

    print("\nAll validation checks passed.\n")
