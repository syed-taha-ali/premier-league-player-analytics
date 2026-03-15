"""
FPL Dataset Explorer
====================
Recursively scans all CSV files, prints columns, and shows samples.
Groups files by logical type to avoid printing hundreds of near-identical player files.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 30)

DATA_ROOT = Path(__file__).parent / "data"

# ── helpers ──────────────────────────────────────────────────────────────────

SEASONS = re.compile(r"^\d{4}-\d{2}$")


def classify_file(path: Path) -> str:
    """Return a logical file-type label for a CSV path."""
    parts = path.parts
    name = path.name

    # root-level aggregated files
    if path.parent == DATA_ROOT:
        return f"ROOT/{name}"

    # season-level root files
    if path.parent.parent == DATA_ROOT and SEASONS.match(path.parent.name):
        return f"SEASON/{name}"

    # gws/ folder
    if "gws" in parts:
        if name == "merged_gw.csv":
            return "GWS/merged_gw.csv"
        if re.match(r"^gw\d+\.csv$", name):
            return "GWS/gw_N.csv"
        if re.match(r"^xP\d+\.csv$", name):
            return "GWS/xP_N.csv"
        return f"GWS/{name}"

    # players/ folder
    if "players" in parts:
        if name == "gw.csv":
            return "PLAYERS/gw.csv"
        if name == "history.csv":
            return "PLAYERS/history.csv"
        return f"PLAYERS/{name}"

    return f"OTHER/{'/'.join(parts[-2:])}"


def scan_files(root: Path) -> dict[str, list[Path]]:
    """Walk the tree and bucket files by type."""
    buckets: dict[str, list[Path]] = defaultdict(list)
    for csv_path in sorted(root.rglob("*.csv")):
        label = classify_file(csv_path)
        buckets[label].append(csv_path)
    return dict(buckets)


def read_safe(path: Path, nrows: int = 3) -> pd.DataFrame | None:
    """Read CSV with fallback encodings; return None on failure."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, nrows=nrows, encoding=enc, low_memory=False)
        except Exception:
            continue
    return None


def season_of(path: Path) -> str:
    for part in path.parts:
        if SEASONS.match(part):
            return part
    return "—"


def representative_files(paths: list[Path]) -> list[Path]:
    """
    For player-level types pick one file per season (+ the very first).
    For season-level types pick the first per season.
    For everything else return all (usually 1).
    """
    by_season: dict[str, Path] = {}
    for p in paths:
        s = season_of(p)
        if s not in by_season:
            by_season[s] = p
    return list(by_season.values())


def hr(char="─", width=100):
    print(char * width)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 100)
    print("  FPL DATASET EXPLORER")
    print("═" * 100)
    print(f"  Root: {DATA_ROOT}\n")

    buckets = scan_files(DATA_ROOT)

    # Summary table first
    print(f"{'FILE TYPE':<35} {'TOTAL FILES':>12} {'SEASONS COVERED'}")
    hr()
    for label, paths in sorted(buckets.items()):
        seasons = sorted({season_of(p) for p in paths} - {"—"})
        season_str = f"{seasons[0]} → {seasons[-1]}" if len(seasons) > 1 else (seasons[0] if seasons else "—")
        print(f"  {label:<33} {len(paths):>12}    {season_str}")
    hr()
    print(f"  {'TOTAL':<33} {sum(len(v) for v in buckets.values()):>12}")
    print()

    # Detailed inspection per file type
    for label, paths in sorted(buckets.items()):
        print("\n" + "═" * 100)
        print(f"  FILE TYPE: {label}  ({len(paths)} files)")
        print("═" * 100)

        reps = representative_files(paths)

        for path in reps:
            rel = path.relative_to(DATA_ROOT.parent)
            print(f"\n  File  : {rel}")
            print(f"  Season: {season_of(path)}")

            df = read_safe(path)
            if df is None:
                print("  [Could not read file]")
                continue

            print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"\n  Columns ({df.shape[1]}):")

            # Print columns in rows of 4
            cols = list(df.columns)
            for i in range(0, len(cols), 4):
                chunk = cols[i:i+4]
                print("    " + "  |  ".join(f"{c:<30}" for c in chunk))

            print(f"\n  Sample (top {min(3, len(df))} rows):")
            print(df.head(3).to_string(index=False))
            hr("-")

    # ── Relational schema proposal ────────────────────────────────────────────
    print("\n" + "═" * 100)
    print("  PROPOSED RELATIONAL SCHEMA")
    print("═" * 100)
    schema = """
  ┌─────────────────────────────────────────────────────────────────────────────────────────┐
  │  dim_season           (season TEXT PK)                                                  │
  │  dim_team             (season, team_id, team_name, short_name, strength_*)              │
  │  dim_player           (player_code INT PK, first_name, second_name, web_name,           │
  │                         element_type/position, birth_date, region)                      │
  │  dim_player_season    (player_code, season FK→dim_season,                               │
  │                         fpl_id INT, team_id FK→dim_team,                                │
  │                         now_cost, status, selected_by_percent,                           │
  │                         total_points, minutes, goals_scored, assists, …)                 │
  │                        [source: players_raw.csv + player_idlist.csv per season]          │
  │  fact_gw_player       (player_code FK, season FK, gw INT,                               │
  │                         fixture_id, opponent_team_id, was_home,                          │
  │                         kickoff_time, minutes, total_points,                             │
  │                         goals_scored, assists, clean_sheets, saves,                      │
  │                         bonus, bps, ict_index, influence, creativity, threat,            │
  │                         xG, xA, xGI, xGC, xP,                                           │
  │                         value, selected, transfers_in, transfers_out)                    │
  │                        [source: gws/merged_gw.csv per season — primary fact table]       │
  │  fact_player_history  (player_code FK, season_name,                                     │
  │                         start_cost, end_cost, total_points, minutes,                     │
  │                         goals_scored, assists, clean_sheets, saves,                      │
  │                         bonus, bps, ict_index, influence, creativity, threat)            │
  │                        [source: players/<name>/history.csv — cross-season summary]       │
  └─────────────────────────────────────────────────────────────────────────────────────────┘

  KEY RELATIONSHIPS
  ─────────────────
  dim_team.team_id      ←→  fact_gw_player.opponent_team_id   (via master_team_list)
  dim_player.player_code ←→ dim_player_season.player_code
  dim_player.player_code ←→ fact_gw_player.player_code
  dim_player.player_code ←→ fact_player_history.player_code
  dim_player_season.fpl_id → gw files: element column           (season-scoped ID)

  NOTE: player_code (players_raw.code) is the stable cross-season identifier.
        fpl_id (players_raw.id / gw files: element) resets each season — do NOT
        join across seasons on fpl_id alone.

  FILES TO EXCLUDE / TREAT AS OUTDATED
  ──────────────────────────────────────
  • cleaned_merged_seasons.csv          — missing 2024-25 and 2025-26
  • cleaned_merged_seasons_team_aggregated.csv — same
  • master_team_list.csv                — missing recent seasons
  → Rebuild these from per-season sources in the pipeline.

  IMPORTANT CAVEATS
  ─────────────────
  • 2016-17 / 2017-18 player folders lack IDs in the folder name.
  • xP column only exists in gw files from 2020-21 onward.
  • 2019-20 ran GW1–GW47 (COVID season).
  • 2025-26 is ongoing (stops at GW24 as of pipeline creation).
  • element_type=5 (AM) appears in some older seasons.
"""
    print(schema)


if __name__ == "__main__":
    main()
