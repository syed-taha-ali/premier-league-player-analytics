"""
ETL entry point. Run with:
    python -m etl.run
from the fpl_analysis/ directory.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .schema import DDL
from .loaders import (
    load_dim_season,
    load_dim_player,
    load_dim_team,
    load_dim_player_season,
    load_fact_player_season_history,
    load_fact_gw_player,
    scan_history,
    build_history_cost_index,
)
from .validate import run_all

DATA_ROOT = Path(__file__).parent.parent / 'data'
DB_PATH   = Path(__file__).parent.parent / 'db' / 'fpl.db'


def create_schema(conn: sqlite3.Connection) -> None:
    # SQLite doesn't support executing multiple statements in execute(),
    # so split on ';' and run each statement individually.
    for stmt in DDL.split(';'):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()


def main() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)

    # Remove existing DB so we start clean on each run
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH.name}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -65536")  # 64 MB page cache
    conn.execute("PRAGMA foreign_keys = ON")

    t0 = time.perf_counter()

    print("Creating schema...")
    create_schema(conn)

    print("Loading dim_season...")
    load_dim_season(conn)
    conn.commit()

    print("Loading dim_player...")
    load_dim_player(conn, DATA_ROOT)
    conn.commit()

    print("Loading dim_team...")
    load_dim_team(conn, DATA_ROOT)
    conn.commit()

    print("Scanning history files (this may take a minute)...")
    history_df    = scan_history(DATA_ROOT)
    history_costs = build_history_cost_index(history_df)
    print(f"  Scanned {len(history_df):,} unique player-season history records")

    print("Loading dim_player_season...")
    load_dim_player_season(conn, DATA_ROOT, history_costs)
    conn.commit()

    print("Loading fact_player_season_history...")
    load_fact_player_season_history(conn, history_df)
    conn.commit()

    print("Loading fact_gw_player...")
    load_fact_gw_player(conn, DATA_ROOT)
    conn.commit()

    # Update total_gws for any season where actual data is less than the seeded 38
    # (i.e., COVID 2019-20 and any ongoing season like 2025-26)
    conn.execute("""
        UPDATE dim_season SET total_gws = (
            SELECT MAX(gw) FROM fact_gw_player f WHERE f.season_id = dim_season.season_id
        )
        WHERE (
            SELECT MAX(gw) FROM fact_gw_player f WHERE f.season_id = dim_season.season_id
        ) IS NOT NULL
    """)
    conn.commit()

    elapsed = time.perf_counter() - t0
    print(f"\nETL complete in {elapsed:.1f}s")

    run_all(conn)

    conn.close()
    size_mb = DB_PATH.stat().st_size / 1_048_576
    print(f"Database written to {DB_PATH}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
