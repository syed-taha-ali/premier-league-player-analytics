"""
Shared utilities for the FPL Analysis dashboard.

All data loaders use @st.cache_data to prevent redundant I/O on page re-runs.
DB access uses a read-only SQLite URI to prevent accidental writes during ETL.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # fpl_analysis/
DB_PATH      = PROJECT_ROOT / "db" / "fpl.db"
PRED_DIR     = PROJECT_ROOT / "outputs" / "predictions"
EDA_DIR      = PROJECT_ROOT / "outputs" / "eda"
MODELS_DIR   = PROJECT_ROOT / "outputs" / "models"
LOG_DIR      = PROJECT_ROOT / "logs" / "monitoring"
TRAIN_DIR    = PROJECT_ROOT / "logs" / "training"

POSITIONS    = ["GK", "DEF", "MID", "FWD"]
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"

# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------

def query_db(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a read-only SQL query and return a DataFrame."""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        try:
            return pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        st.error(f"Database unavailable — ETL may be running. Try again in a moment. ({e})")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Player / team name loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_player_names() -> dict[int, str]:
    """Return {player_code: web_name} mapping from dim_player."""
    df = query_db("SELECT player_code, web_name FROM dim_player")
    if df.empty:
        return {}
    return dict(zip(df["player_code"], df["web_name"]))


@st.cache_data
def load_team_names() -> dict[int, str]:
    """Return {team_sk: team_name} mapping from dim_team."""
    df = query_db("SELECT team_sk, team_name FROM dim_team")
    if df.empty:
        return {}
    return dict(zip(df["team_sk"], df["team_name"]))


@st.cache_data
def load_season_list() -> list[dict]:
    """Return list of {season_id, season_label} dicts from dim_season, newest first."""
    df = query_db("SELECT season_id, season_label FROM dim_season ORDER BY season_id DESC")
    if df.empty:
        return []
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# Prediction loaders
# ---------------------------------------------------------------------------

def list_available_gws() -> list[tuple[int, int]]:
    """
    Return sorted list of (gw, season_id) tuples from prediction CSVs in PRED_DIR,
    newest first (descending by season_id then gw).
    """
    if not PRED_DIR.exists():
        return []
    result = []
    pattern = re.compile(r"gw(\d+)_s(\d+)_predictions\.csv")
    for f in PRED_DIR.glob("gw*_s*_predictions.csv"):
        m = pattern.match(f.name)
        if m:
            result.append((int(m.group(1)), int(m.group(2))))
    result.sort(key=lambda t: (t[1], t[0]), reverse=True)
    return result


@st.cache_data
def load_predictions(gw: int, season_id: int) -> pd.DataFrame:
    """
    Load prediction CSV for the given GW/season and enrich with:
    - player web_name (from dim_player)
    - opponent team name (from fact_gw_player JOIN dim_team)
    """
    path = PRED_DIR / f"gw{gw}_s{season_id}_predictions.csv"
    if not path.exists():
        st.warning(
            f"No prediction file found for GW {gw} season {season_id}. "
            "Run run_gw.py to generate predictions."
        )
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Join player names
    player_names = load_player_names()
    if player_names and "player_code" in df.columns:
        df["web_name"] = df["player_code"].map(player_names).fillna("Unknown")

    # Join opponent team names via DB
    opp_df = query_db(
        """
        SELECT DISTINCT
            f.season_id, f.gw, f.fixture_id, f.player_code,
            dt.team_name AS opponent_team,
            CASE WHEN f.was_home = 1 THEN 'H' ELSE 'A' END AS home_away
        FROM fact_gw_player f
        JOIN dim_team dt ON dt.team_sk = f.opponent_team_sk
        WHERE f.season_id = ? AND f.gw = ?
        """,
        params=(season_id, gw),
    )
    if not opp_df.empty and "fixture_id" in df.columns:
        df = df.merge(
            opp_df[["fixture_id", "player_code", "opponent_team", "home_away"]],
            on=["fixture_id", "player_code"],
            how="left",
        )

    # Join ownership % (selected is stored as a percentage in fact_gw_player)
    own_df = query_db(
        """
        SELECT player_code, gw, season_id, selected AS ownership_pct
        FROM fact_gw_player
        WHERE season_id = ? AND gw = ? AND selected IS NOT NULL
        """,
        params=(season_id, gw),
    )
    if not own_df.empty:
        df = df.merge(
            own_df[["player_code", "ownership_pct"]],
            on="player_code",
            how="left",
        )

    # Add team name from dim_team
    team_names = load_team_names()
    if team_names and "team_sk" in df.columns:
        df["team"] = df["team_sk"].map(team_names).fillna("Unknown")

    return df


# ---------------------------------------------------------------------------
# Monitoring / training loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_monitoring_log() -> pd.DataFrame:
    """Load logs/monitoring/monitoring_log.csv."""
    path = LOG_DIR / "monitoring_log.csv"
    if not path.exists():
        st.info("No monitoring data yet. Run run_gw.py after the first GW.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_cv_metrics() -> pd.DataFrame:
    """Load logs/training/cv_metrics_all.csv."""
    path = TRAIN_DIR / "cv_metrics_all.csv"
    if not path.exists():
        st.warning("CV metrics file not found. Run python -m ml.evaluate first.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_fdr_calendar(season_id: int) -> pd.DataFrame:
    """
    Build a team × GW FDR DataFrame for the given season from the feature matrix.

    Returns columns: team_sk, gw, opponent_season_rank, fdr, team_name, opponent_team_name.
    One row per (team_sk, gw) — takes the first player row since all players on the same
    team in the same fixture share the same opponent_season_rank.
    """
    # Load any position's feature matrix (MID has best coverage)
    path = FEATURES_DIR / "feature_matrix_MID.parquet"
    if not path.exists():
        st.warning("Feature matrix cache missing. Clear outputs/features/ and re-run the pipeline.")
        return pd.DataFrame()

    fm = pd.read_parquet(path)
    fm_season = fm[fm["season_id"] == season_id][
        ["team_sk", "gw", "opponent_season_rank", "fixture_id"]
    ].copy()

    # One row per (team_sk, gw) — use first occurrence per fixture
    fm_dedup = (
        fm_season.sort_values("fixture_id")
        .groupby(["team_sk", "gw"], as_index=False)
        .first()
    )

    # FDR from opponent_season_rank: rank 1–6 = FDR 5 (hardest), 19–20 = FDR 1 (easiest)
    def _rank_to_fdr(rank: float) -> int:
        if rank <= 6:
            return 5
        if rank <= 10:
            return 4
        if rank <= 14:
            return 3
        if rank <= 17:
            return 2
        return 1

    fm_dedup["fdr"] = fm_dedup["opponent_season_rank"].apply(_rank_to_fdr)

    # Join own team names
    team_names = load_team_names()
    fm_dedup["team_name"] = fm_dedup["team_sk"].map(team_names).fillna("Unknown")

    # Join opponent team names via DB
    opp_map = query_db(
        """
        SELECT DISTINCT f.team_sk, f.gw, f.season_id, dt.team_name AS opponent_team_name
        FROM fact_gw_player f
        JOIN dim_team dt ON dt.team_sk = f.opponent_team_sk
        WHERE f.season_id = ?
        """,
        params=(season_id,),
    )
    if not opp_map.empty:
        fm_dedup = fm_dedup.merge(
            opp_map[["team_sk", "gw", "opponent_team_name"]].drop_duplicates(["team_sk", "gw"]),
            on=["team_sk", "gw"],
            how="left",
        )

    return fm_dedup


@st.cache_data
def load_oof(position: str) -> pd.DataFrame:
    """
    Load OOF predictions parquet for the given position and join player web_names.
    Returns cv_preds_{position}.parquet with an added 'web_name' column.
    """
    path = TRAIN_DIR / f"cv_preds_{position}.parquet"
    if not path.exists():
        st.warning(
            f"OOF predictions not found for {position}. "
            "Run python -m ml.evaluate to generate them."
        )
        return pd.DataFrame()
    df = pd.read_parquet(path)
    player_names = load_player_names()
    if player_names and "player_code" in df.columns:
        df["web_name"] = df["player_code"].map(player_names).fillna("Unknown")
    return df
