"""
Page 6 — Database Explorer

Browse the FPL SQLite database via 20 preset query templates (Player / Team / Gameweek /
Advanced), a raw table browser, and a free-form SQL interface.

Data scope: FPL fantasy data only — points, player stats per GW, team rosters per season.
Official Premier League standings are not stored. "Who won the league" is answered via
aggregate FPL points across the squad (Template 3).
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import LOG_DIR, POSITIONS, query_db

st.set_page_config(layout="wide", page_title="FPL Analysis — Database Explorer")

st.title("Database Explorer")
st.caption(
    "Preset query templates, raw table browser, and free-form SQL. "
    "All queries are read-only — no data can be modified from the dashboard."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_TABLES = [
    "dim_player",
    "dim_team",
    "dim_season",
    "dim_player_season",
    "fact_gw_player",
    "fact_player_season_history",
]

XG_ERA_MIN_SEASON = 7  # 2022-23 onwards


def _season_options() -> list[dict]:
    """Return list of {season_id, season_label} dicts, newest first."""
    df = query_db("SELECT season_id, season_label FROM dim_season ORDER BY season_id DESC")
    return df.to_dict("records") if not df.empty else []


def _team_options(season_id: int) -> list[str]:
    """Return sorted list of team names for the given season."""
    df = query_db(
        "SELECT DISTINCT team_name FROM dim_team WHERE season_id = ? ORDER BY team_name",
        params=(season_id,),
    )
    return df["team_name"].tolist() if not df.empty else []


def _player_search(label: str, key: str) -> int | None:
    """
    Partial-name search: text_input → DB query → selectbox → returns player_code or None.
    """
    q = st.text_input(label, key=key, placeholder="e.g. Salah")
    if not q:
        return None
    matches = query_db(
        "SELECT player_code, web_name FROM dim_player WHERE LOWER(web_name) LIKE LOWER(?) LIMIT 20",
        params=(f"%{q}%",),
    )
    if matches.empty:
        st.info(f"No players found matching '{q}'.")
        return None
    options = dict(zip(matches["web_name"], matches["player_code"]))
    chosen_name = st.selectbox("Select player", list(options.keys()), key=f"{key}_sel")
    return options[chosen_name]


def _result_block(df: pd.DataFrame, filename: str = "result.csv") -> None:
    """Show dataframe + row count + download button."""
    if df.empty:
        st.info("No results returned.")
        return
    st.dataframe(df, use_container_width=True)
    st.caption(f"{len(df):,} rows")
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        file_name=filename,
        mime="text/csv",
    )


def _xg_warning(season_id: int) -> None:
    """Display a Streamlit warning if the selected season pre-dates xG availability."""
    if season_id < XG_ERA_MIN_SEASON:
        st.warning("xG stats are not available before 2022-23 (season 7). xG columns will be NULL.")


def _pos_placeholder(positions: list[str]) -> str:
    """Return SQL-safe IN placeholder string for a list of positions."""
    return ", ".join(f"'{p}'" for p in positions)


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, list[str]] = {
    "Player": [
        "1 — Team Season Roster",
        "2 — Top FPL Scorers by Season",
        "4 — Player Career Stats",
        "6 — Player Head-to-Head",
        "13 — Haul Hunters",
        "14 — Current Form Table",
        "15 — Home vs Away Splits",
        "16 — Attacking Returns",
    ],
    "Team": [
        "3 — Team Season Summary",
        "11 — Defensive Record",
    ],
    "Gameweek": [
        "5 — GW Results",
        "10 — Transfer & Ownership Trends",
        "17 — Double Gameweek Finder",
    ],
    "Advanced": [
        "7 — xG / Advanced Stats Leaders",
        "8 — Price Movers",
        "9 — Reliable Starters",
        "12 — Bonus Point Leaders",
        "18 — GK-Specific Stats",
        "19 — Suspension Risk",
        "20 — Season History Explorer",
    ],
}

# ---------------------------------------------------------------------------
# Schema reference (collapsible)
# ---------------------------------------------------------------------------

with st.expander("DB Schema Reference"):
    st.markdown(
        """
| Table | Grain | Key columns |
|-------|-------|-------------|
| `dim_season` | season | `season_id`, `season_label`, `total_gws` |
| `dim_player` | player | `player_code`, `web_name`, `first_name`, `second_name` |
| `dim_team` | team × season | `team_sk`, `team_name`, `season_id` |
| `dim_player_season` | player × season | `player_code`, `season_id`, `team_sk`, `total_points`, `position_label` |
| `fact_gw_player` | player × fixture | `player_code`, `season_id`, `gw`, `fixture_id`, `total_points`, `minutes`, `was_home` |
| `fact_player_season_history` | player × prior season | `player_code`, `season_id`, `total_points`, `start_cost`, `end_cost` |

Seasons: 1 = 2016-17 … 10 = 2025-26. xG data available from season 7 (2022-23) onwards.
All costs stored as £0.1m units — divide by 10 for display.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Section A — Preset Queries
# ---------------------------------------------------------------------------

st.subheader("Preset Queries")

col_cat, col_tpl = st.columns([1, 3])
with col_cat:
    category = st.selectbox("Category", list(TEMPLATES.keys()))
with col_tpl:
    template = st.selectbox("Query template", TEMPLATES[category])

st.divider()

seasons = _season_options()
season_map = {s["season_label"]: s["season_id"] for s in seasons}
season_labels = [s["season_label"] for s in seasons]
pos_options = ["All"] + POSITIONS

# ---- Template 1: Team Season Roster ----------------------------------------
if template.startswith("1"):
    st.markdown("**Team Season Roster** — Which players were at [team] in [season]?")
    c1, c2, c3 = st.columns(3)
    with c1:
        season_lbl = st.selectbox("Season", season_labels, key="t1_season")
        season_id  = season_map[season_lbl]
    with c2:
        team_opts  = _team_options(season_id)
        team_name  = st.selectbox("Team", team_opts, key="t1_team") if team_opts else None
    with c3:
        pos_sel    = st.multiselect("Position", POSITIONS, default=POSITIONS, key="t1_pos")

    if team_name and pos_sel and st.button("Run", key="t1_run"):
        pos_ph = _pos_placeholder(pos_sel)
        df = query_db(
            f"""
            SELECT
                dp.web_name         AS player,
                dps.position_label  AS position,
                dps.total_points    AS season_pts,
                ROUND(dps.start_cost / 10.0, 1) AS start_price_m,
                ROUND(dps.end_cost   / 10.0, 1) AS end_price_m,
                dps.goals_scored,
                dps.assists,
                dps.clean_sheets,
                dps.bonus
            FROM dim_player_season dps
            JOIN dim_player dp ON dp.player_code = dps.player_code
            JOIN dim_team   dt ON dt.team_sk = dps.team_sk AND dt.season_id = dps.season_id
            WHERE dt.team_name  = ?
              AND dps.season_id = ?
              AND dps.position_label IN ({pos_ph})
            ORDER BY dps.total_points DESC
            """,
            params=(team_name, season_id),
        )
        _result_block(df, f"roster_{team_name}_{season_lbl}.csv")

# ---- Template 2: Top FPL Scorers -------------------------------------------
elif template.startswith("2"):
    st.markdown("**Top FPL Scorers** — Who had the highest FPL points in [season]?")
    c1, c2, c3 = st.columns(3)
    with c1:
        season_lbl = st.selectbox("Season", season_labels, key="t2_season")
        season_id  = season_map[season_lbl]
    with c2:
        pos_sel    = st.selectbox("Position", pos_options, key="t2_pos")
    with c3:
        top_n      = st.number_input("Show top N", min_value=1, max_value=200, value=20, key="t2_n")

    if st.button("Run", key="t2_run"):
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        params     = (season_id, pos_sel) if pos_sel != "All" else (season_id,)
        df = query_db(
            f"""
            SELECT
                dp.web_name         AS player,
                dt.team_name        AS team,
                dps.position_label  AS position,
                dps.total_points    AS season_pts,
                dps.goals_scored,
                dps.assists,
                ROUND(dps.start_cost / 10.0, 1) AS start_price_m
            FROM dim_player_season dps
            JOIN dim_player dp ON dp.player_code = dps.player_code
            JOIN dim_team   dt ON dt.team_sk = dps.team_sk AND dt.season_id = dps.season_id
            WHERE dps.season_id = ? {pos_clause}
            ORDER BY dps.total_points DESC
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        _result_block(df, f"top_scorers_{season_lbl}.csv")

# ---- Template 3: Team Season Summary ----------------------------------------
elif template.startswith("3"):
    st.markdown(
        "**Team Season Summary** — Aggregate FPL stats per team. "
        "Sorted by total FPL points — not official Premier League standing."
    )
    c1, c2 = st.columns(2)
    with c1:
        season_lbl = st.selectbox("Season", season_labels, key="t3_season")
        season_id  = season_map[season_lbl]
    with c2:
        sort_opts  = {"Total FPL pts": "total_fpl_pts", "Goals scored": "total_goals",
                      "Squad size": "squad_size"}
        sort_by    = st.selectbox("Sort by", list(sort_opts.keys()), key="t3_sort")
        sort_col   = sort_opts[sort_by]

    if st.button("Run", key="t3_run"):
        df = query_db(
            f"""
            SELECT
                dt.team_name,
                COUNT(DISTINCT dps.player_code) AS squad_size,
                SUM(dps.total_points)           AS total_fpl_pts,
                SUM(dps.goals_scored)           AS total_goals,
                SUM(dps.assists)                AS total_assists
            FROM dim_player_season dps
            JOIN dim_team dt ON dt.team_sk = dps.team_sk AND dt.season_id = dps.season_id
            WHERE dps.season_id = ?
            GROUP BY dt.team_name
            ORDER BY {sort_col} DESC
            """,
            params=(season_id,),
        )
        _result_block(df, f"team_summary_{season_lbl}.csv")
        st.caption("Sorted by total FPL points across all squad members — not official Premier League standing.")

# ---- Template 4: Player Career Stats ----------------------------------------
elif template.startswith("4"):
    st.markdown("**Player Career Stats** — Show a player's stats across all seasons.")
    player_code = _player_search("Player name", "t4_player")

    if player_code and st.button("Run", key="t4_run"):
        df = query_db(
            """
            SELECT
                ds.season_label     AS season,
                dt.team_name        AS team,
                dps.position_label  AS position,
                dps.total_points    AS season_pts,
                dps.goals_scored,
                dps.assists,
                dps.clean_sheets,
                ROUND(dps.start_cost / 10.0, 1) AS start_price_m,
                ROUND(dps.end_cost   / 10.0, 1) AS end_price_m
            FROM dim_player_season dps
            JOIN dim_season ds ON ds.season_id = dps.season_id
            JOIN dim_team   dt ON dt.team_sk   = dps.team_sk AND dt.season_id = dps.season_id
            WHERE dps.player_code = ?
            ORDER BY dps.season_id
            """,
            params=(player_code,),
        )
        _result_block(df, "player_career.csv")
        if not df.empty:
            fig = px.bar(df, x="season", y="season_pts", color="team",
                         title="Season points by year", labels={"season_pts": "FPL pts"})
            st.plotly_chart(fig, use_container_width=True)

# ---- Template 5: GW Results --------------------------------------------------
elif template.startswith("5"):
    st.markdown("**GW Results** — All player scores for a given gameweek.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t5_season")
        season_id   = season_map[season_lbl]
    with c2:
        max_gw_row  = query_db("SELECT MAX(gw) AS mg FROM fact_gw_player WHERE season_id = ?",
                               params=(season_id,))
        max_gw      = int(max_gw_row["mg"].iloc[0]) if not max_gw_row.empty else 38
        gw_num      = st.number_input("Gameweek", min_value=1, max_value=max_gw, value=min(30, max_gw), key="t5_gw")
    with c3:
        pos_sel     = st.multiselect("Position", POSITIONS, default=POSITIONS, key="t5_pos")
    with c4:
        min_mins    = st.number_input("Min minutes", min_value=0, max_value=90, value=0, key="t5_mins")

    if pos_sel and st.button("Run", key="t5_run"):
        pos_ph = _pos_placeholder(pos_sel)
        df = query_db(
            f"""
            SELECT
                dp.web_name         AS player,
                dps.position_label  AS position,
                dt.team_name        AS team,
                f.total_points      AS pts,
                f.minutes,
                f.goals_scored,
                f.assists,
                f.clean_sheets,
                f.bonus,
                CASE WHEN f.was_home = 1 THEN 'H' ELSE 'A' END AS venue
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ?
              AND f.gw = ?
              AND f.minutes >= ?
              AND dps.position_label IN ({pos_ph})
            ORDER BY f.total_points DESC
            """,
            params=(season_id, int(gw_num), int(min_mins)),
        )
        _result_block(df, f"gw{int(gw_num)}_{season_lbl}_results.csv")

# ---- Template 6: Player Head-to-Head -----------------------------------------
elif template.startswith("6"):
    st.markdown("**Player Head-to-Head** — Compare two players across seasons.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Player A**")
        code_a = _player_search("Player A name", "t6_a")
    with c2:
        st.markdown("**Player B**")
        code_b = _player_search("Player B name", "t6_b")

    if code_a and code_b and st.button("Compare", key="t6_run"):
        def _career(pcode):
            """Fetch season-by-season stats for one player_code."""
            return query_db(
                """
                SELECT ds.season_label AS season, dps.total_points AS season_pts,
                       dps.goals_scored, dps.assists,
                       dp.web_name AS player
                FROM dim_player_season dps
                JOIN dim_season ds ON ds.season_id = dps.season_id
                JOIN dim_player dp ON dp.player_code = dps.player_code
                WHERE dps.player_code = ?
                ORDER BY dps.season_id
                """,
                params=(pcode,),
            )

        df_a = _career(code_a)
        df_b = _career(code_b)
        col_a, col_b = st.columns(2)
        with col_a:
            if not df_a.empty:
                st.markdown(f"**{df_a['player'].iloc[0]}**")
                _result_block(df_a.drop(columns=["player"]), "player_a_career.csv")
        with col_b:
            if not df_b.empty:
                st.markdown(f"**{df_b['player'].iloc[0]}**")
                _result_block(df_b.drop(columns=["player"]), "player_b_career.csv")

        if not df_a.empty and not df_b.empty:
            combined = pd.concat([df_a, df_b])
            fig = px.bar(combined, x="season", y="season_pts", color="player",
                         barmode="group", title="Season points comparison",
                         labels={"season_pts": "FPL pts"})
            st.plotly_chart(fig, use_container_width=True)

# ---- Template 7: xG Leaders --------------------------------------------------
elif template.startswith("7"):
    st.markdown("**xG / Advanced Stats Leaders** — xG era only (2022-23 onwards).")
    xg_seasons  = [s for s in season_labels if season_map[s] >= XG_ERA_MIN_SEASON]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        season_lbl  = st.selectbox("Season", xg_seasons, key="t7_season")
        season_id   = season_map[season_lbl]
    with c2:
        metric      = st.selectbox("Metric", ["xG", "xA", "xGI", "xGC"], key="t7_metric")
    with c3:
        pos_sel     = st.selectbox("Position", pos_options, key="t7_pos")
    with c4:
        top_n       = st.number_input("Top N", min_value=1, max_value=200, value=20, key="t7_n")
    with c5:
        per90       = st.checkbox("Per 90 mins", key="t7_per90")

    metric_col_map = {
        "xG":  "expected_goals",
        "xA":  "expected_assists",
        "xGI": "expected_goal_involvements",
        "xGC": "expected_goals_conceded",
    }
    mc = metric_col_map[metric]

    def _xg_expr(col, per_90, alias):
        """Return the SQL expression for a cumulative or per-90 xG stat."""
        if per_90:
            return f"ROUND(SUM(f.{col}) / NULLIF(SUM(f.minutes) / 90.0, 0), 3) AS {alias}"
        return f"ROUND(SUM(f.{col}), 2) AS {alias}"

    if st.button("Run", key="t7_run"):
        _xg_warning(season_id)
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        params     = (season_id, pos_sel) if pos_sel != "All" else (season_id,)
        expr       = _xg_expr(mc, per90, "metric_value")
        df = query_db(
            f"""
            SELECT
                dp.web_name             AS player,
                dt.team_name            AS team,
                dps.position_label      AS position,
                {expr},
                SUM(f.goals_scored)     AS goals,
                SUM(f.assists)          AS assists,
                SUM(f.minutes)          AS minutes
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ? AND f.minutes > 0 {pos_clause}
            GROUP BY f.player_code
            ORDER BY metric_value DESC
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        if not df.empty:
            df = df.rename(columns={"metric_value": f"{metric}{'_p90' if per90 else ''}"})
        _result_block(df, f"xg_leaders_{season_lbl}.csv")

# ---- Template 8: Price Movers ------------------------------------------------
elif template.startswith("8"):
    st.markdown("**Price Movers** — Biggest price changes in a season.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t8_season")
        season_id   = season_map[season_lbl]
    with c2:
        direction   = st.selectbox("Direction", ["Biggest risers", "Biggest fallers", "Both"], key="t8_dir")
    with c3:
        pos_sel     = st.selectbox("Position", pos_options, key="t8_pos")
    with c4:
        top_n       = st.number_input("Top N", min_value=1, max_value=200, value=20, key="t8_n")

    if st.button("Run", key="t8_run"):
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        sort_dir   = "DESC" if direction == "Biggest risers" else "ASC"
        params     = (season_id, pos_sel) if pos_sel != "All" else (season_id,)
        df = query_db(
            f"""
            SELECT
                dp.web_name         AS player,
                dt.team_name        AS team,
                dps.position_label  AS position,
                ROUND(dps.start_cost / 10.0, 1)                       AS start_price_m,
                ROUND(dps.end_cost   / 10.0, 1)                       AS end_price_m,
                ROUND((dps.end_cost - dps.start_cost) / 10.0, 1)      AS price_change_m,
                dps.total_points                                        AS season_pts
            FROM dim_player_season dps
            JOIN dim_player dp ON dp.player_code = dps.player_code
            JOIN dim_team   dt ON dt.team_sk = dps.team_sk AND dt.season_id = dps.season_id
            WHERE dps.season_id = ?
              AND dps.start_cost IS NOT NULL
              AND dps.end_cost IS NOT NULL
              {pos_clause}
            ORDER BY price_change_m {sort_dir}
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        if direction == "Both":
            df = df.reindex(
                df["price_change_m"].abs().sort_values(ascending=False).index
            )
        _result_block(df, f"price_movers_{season_lbl}.csv")
        if not df.empty:
            fig = px.bar(df, x="player", y="price_change_m", color="position",
                         title="Price change (£m)", labels={"price_change_m": "Change (£m)"})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

# ---- Template 9: Reliable Starters -------------------------------------------
elif template.startswith("9"):
    st.markdown("**Reliable Starters** — Minutes leaders — rotation risk signal.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t9_season")
        season_id   = season_map[season_lbl]
    with c2:
        pos_sel     = st.selectbox("Position", pos_options, key="t9_pos")
    with c3:
        min_apps    = st.number_input("Min appearances", min_value=1, max_value=50, value=10, key="t9_apps")
    with c4:
        top_n       = st.number_input("Top N", min_value=1, max_value=200, value=20, key="t9_n")

    if st.button("Run", key="t9_run"):
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        params     = (season_id, pos_sel) if pos_sel != "All" else (season_id,)
        df = query_db(
            f"""
            SELECT
                dp.web_name                                     AS player,
                dt.team_name                                    AS team,
                dps.position_label                              AS position,
                SUM(f.minutes)                                  AS total_minutes,
                COUNT(DISTINCT f.fixture_id)                    AS appearances,
                ROUND(AVG(f.minutes), 1)                        AS avg_mins_per_gw,
                ROUND(SUM(f.total_points) * 1.0
                      / COUNT(DISTINCT f.fixture_id), 2)        AS pts_per_app,
                dps.total_points                                AS season_pts
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ? AND f.minutes > 0 {pos_clause}
            GROUP BY f.player_code
            HAVING appearances >= {int(min_apps)}
            ORDER BY total_minutes DESC
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        _result_block(df, f"starters_{season_lbl}.csv")

# ---- Template 10: Transfer & Ownership Trends --------------------------------
elif template.startswith("10"):
    st.markdown("**Transfer & Ownership Trends** — Most transferred in/out in a GW.")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t10_season")
        season_id   = season_map[season_lbl]
    with c2:
        max_gw_row  = query_db("SELECT MAX(gw) AS mg FROM fact_gw_player WHERE season_id = ?",
                               params=(season_id,))
        max_gw      = int(max_gw_row["mg"].iloc[0]) if not max_gw_row.empty else 38
        gw_num      = st.number_input("Gameweek", min_value=1, max_value=max_gw, value=min(30, max_gw), key="t10_gw")
    with c3:
        metric      = st.selectbox("Metric", ["Transfers in", "Transfers out", "Ownership"], key="t10_metric")
    with c4:
        pos_sel     = st.selectbox("Position", pos_options, key="t10_pos")
    with c5:
        top_n       = st.number_input("Top N", min_value=1, max_value=200, value=20, key="t10_n")

    metric_col_map = {"Transfers in": "transfers_in", "Transfers out": "transfers_out",
                      "Ownership": "selected"}

    if st.button("Run", key="t10_run"):
        mc = metric_col_map[metric]
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        params     = (season_id, int(gw_num), pos_sel) if pos_sel != "All" else (season_id, int(gw_num))
        df = query_db(
            f"""
            SELECT
                dp.web_name         AS player,
                dt.team_name        AS team,
                dps.position_label  AS position,
                f.transfers_in,
                f.transfers_out,
                f.selected          AS ownership_pct,
                f.total_points      AS pts_that_gw,
                ROUND(f.value / 10.0, 1) AS price_m
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ? AND f.gw = ? {pos_clause}
            ORDER BY f.{mc} DESC
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        _result_block(df, f"transfers_gw{int(gw_num)}_{season_lbl}.csv")

# ---- Template 11: Defensive Record -------------------------------------------
elif template.startswith("11"):
    st.markdown(
        "**Defensive Record** — Team clean sheet rates. "
        "Uses GK rows as a proxy — avoids the known per-player `goals_conceded` bias."
    )
    c1, = st.columns(1)
    season_lbl  = st.selectbox("Season", season_labels, key="t11_season")
    season_id   = season_map[season_lbl]

    if st.button("Run", key="t11_run"):
        df = query_db(
            """
            SELECT
                dt.team_name,
                COUNT(DISTINCT f.fixture_id)                        AS fixtures_played,
                SUM(CASE WHEN f.clean_sheets = 1 THEN 1 ELSE 0 END) AS clean_sheets,
                ROUND(
                    100.0 * SUM(CASE WHEN f.clean_sheets = 1 THEN 1 ELSE 0 END)
                    / COUNT(DISTINCT f.fixture_id), 1
                )                                                   AS cs_pct
            FROM fact_gw_player f
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team dt ON dt.team_sk = f.team_sk
            WHERE f.season_id = ?
              AND dps.position_label = 'GK'
              AND f.minutes >= 60
            GROUP BY dt.team_name
            ORDER BY cs_pct DESC
            """,
            params=(season_id,),
        )
        _result_block(df, f"defensive_record_{season_lbl}.csv")
        st.caption(
            "Team-level goals conceded is derived via GK clean sheet rates. "
            "The player-level `goals_conceded` column is time-on-pitch scoped and unreliable for team totals."
        )

# ---- Template 12: Bonus Point Leaders ----------------------------------------
elif template.startswith("12"):
    st.markdown("**Bonus Point Leaders** — Who accumulated the most bonus points?")
    c1, c2, c3 = st.columns(3)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t12_season")
        season_id   = season_map[season_lbl]
    with c2:
        pos_sel     = st.selectbox("Position", pos_options, key="t12_pos")
    with c3:
        top_n       = st.number_input("Top N", min_value=1, max_value=200, value=20, key="t12_n")

    if st.button("Run", key="t12_run"):
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        params     = (season_id, pos_sel) if pos_sel != "All" else (season_id,)
        df = query_db(
            f"""
            SELECT
                dp.web_name                             AS player,
                dt.team_name                            AS team,
                dps.position_label                      AS position,
                SUM(f.bonus)                            AS total_bonus,
                ROUND(AVG(f.bonus), 2)                  AS avg_bonus_per_gw,
                SUM(f.bps)                              AS total_bps,
                COUNT(DISTINCT f.fixture_id)            AS appearances,
                dps.total_points                        AS season_pts,
                ROUND(100.0 * SUM(f.bonus)
                      / NULLIF(dps.total_points, 0), 1) AS bonus_pct_of_pts
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ? AND f.minutes > 0 {pos_clause}
            GROUP BY f.player_code
            ORDER BY total_bonus DESC
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        _result_block(df, f"bonus_leaders_{season_lbl}.csv")

# ---- Template 13: Haul Hunters -----------------------------------------------
elif template.startswith("13"):
    st.markdown("**Haul Hunters** — Which players score big points most often?")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        season_opts  = ["All xG era"] + season_labels
        season_lbl   = st.selectbox("Season", season_opts, key="t13_season")
    with c2:
        pos_sel      = st.selectbox("Position", pos_options, key="t13_pos")
    with c3:
        threshold    = st.number_input("Haul threshold (pts)", min_value=6, max_value=20, value=10, key="t13_thr")
    with c4:
        top_n        = st.number_input("Top N", min_value=1, max_value=200, value=20, key="t13_n")

    if st.button("Run", key="t13_run"):
        if season_lbl == "All xG era":
            season_clause = "AND f.season_id >= 7"
            params        = ()
        else:
            season_id     = season_map[season_lbl]
            season_clause = "AND f.season_id = ?"
            params        = (season_id,)

        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        if pos_sel != "All":
            params += (pos_sel,)

        df = query_db(
            f"""
            SELECT
                dp.web_name                                                 AS player,
                dt.team_name                                                AS team,
                dps.position_label                                          AS position,
                COUNT(DISTINCT f.fixture_id)                                AS appearances,
                SUM(CASE WHEN f.total_points >= {int(threshold)} THEN 1 ELSE 0 END) AS hauls,
                ROUND(
                    100.0 * SUM(CASE WHEN f.total_points >= {int(threshold)} THEN 1 ELSE 0 END)
                    / COUNT(DISTINCT f.fixture_id), 1
                )                                                           AS haul_rate_pct,
                ROUND(AVG(f.total_points), 2)                               AS avg_pts,
                MAX(f.total_points)                                         AS best_gw
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.minutes > 0 {season_clause} {pos_clause}
            GROUP BY f.player_code
            HAVING appearances >= 5
            ORDER BY hauls DESC, haul_rate_pct DESC
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        _result_block(df, "haul_hunters.csv")
        if not df.empty:
            fig = px.bar(df.head(20), x="player", y="hauls", color="position",
                         hover_data=["haul_rate_pct", "avg_pts"],
                         title=f"Players with most {int(threshold)}+ point GWs",
                         labels={"hauls": "Haul count"})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

# ---- Template 14: Current Form Table -----------------------------------------
elif template.startswith("14"):
    st.markdown("**Current Form Table** — Who is in form over recent gameweeks?")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t14_season")
        season_id   = season_map[season_lbl]
    with c2:
        n_gws       = st.number_input("Last N GWs", min_value=1, max_value=10, value=5, key="t14_ngws")
    with c3:
        pos_sel     = st.selectbox("Position", pos_options, key="t14_pos")
    with c4:
        min_mins    = st.number_input("Min mins/GW", min_value=0, max_value=90, value=45, key="t14_mins")
    with c5:
        top_n       = st.number_input("Top N", min_value=1, max_value=200, value=30, key="t14_n")

    if st.button("Run", key="t14_run"):
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        params     = (season_id, season_id, int(n_gws), int(min_mins), pos_sel) \
                     if pos_sel != "All" else \
                     (season_id, season_id, int(n_gws), int(min_mins))
        df = query_db(
            f"""
            WITH max_gw AS (
                SELECT MAX(gw) AS mgw FROM fact_gw_player WHERE season_id = ?
            )
            SELECT
                dp.web_name                     AS player,
                dt.team_name                    AS team,
                dps.position_label              AS position,
                COUNT(DISTINCT f.fixture_id)    AS gws_played,
                SUM(f.total_points)             AS total_pts,
                ROUND(AVG(f.total_points), 2)   AS avg_pts,
                MAX(f.total_points)             AS best_gw,
                ROUND(AVG(f.minutes), 0)        AS avg_minutes,
                ROUND(MAX(f.value) / 10.0, 1)  AS current_price_m
            FROM fact_gw_player f, max_gw
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ?
              AND f.gw > max_gw.mgw - ?
              AND f.minutes >= ?
              {pos_clause}
            GROUP BY f.player_code
            ORDER BY total_pts DESC
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        _result_block(df, f"form_table_{season_lbl}.csv")

# ---- Template 15: Home vs Away Splits ----------------------------------------
elif template.startswith("15"):
    st.markdown("**Home vs Away Splits** — How does a player perform at home vs away?")
    player_code = _player_search("Player name", "t15_player")
    season_opts = ["All seasons"] + season_labels
    season_lbl  = st.selectbox("Season", season_opts, key="t15_season")

    if player_code and st.button("Run", key="t15_run"):
        if season_lbl == "All seasons":
            season_clause = ""
            params        = (player_code,)
        else:
            season_id     = season_map[season_lbl]
            season_clause = "AND f.season_id = ?"
            params        = (player_code, season_id)

        df = query_db(
            f"""
            SELECT
                CASE WHEN f.was_home = 1 THEN 'Home' ELSE 'Away' END AS venue,
                COUNT(DISTINCT f.fixture_id)                          AS appearances,
                SUM(f.total_points)                                   AS total_pts,
                ROUND(AVG(f.total_points), 2)                         AS avg_pts,
                SUM(f.goals_scored)                                   AS goals,
                SUM(f.assists)                                        AS assists,
                SUM(f.clean_sheets)                                   AS clean_sheets,
                ROUND(AVG(f.minutes), 0)                              AS avg_minutes,
                ROUND(SUM(f.expected_goals), 2)                       AS xg,
                ROUND(SUM(f.expected_assists), 2)                     AS xa
            FROM fact_gw_player f
            WHERE f.player_code = ? AND f.minutes > 0 {season_clause}
            GROUP BY venue
            """,
            params=params,
        )
        if not df.empty:
            col_a, col_b = st.columns(2)
            for col, (_, row) in zip([col_a, col_b], df.iterrows()):
                col.metric(row["venue"], f"{row['avg_pts']:.2f} avg pts",
                           delta=f"{int(row['appearances'])} apps")
            _result_block(df, "home_away_splits.csv")
            # Seasonal breakdown
            df_seas = query_db(
                f"""
                SELECT
                    ds.season_label AS season,
                    CASE WHEN f.was_home = 1 THEN 'Home' ELSE 'Away' END AS venue,
                    ROUND(AVG(f.total_points), 2) AS avg_pts
                FROM fact_gw_player f
                JOIN dim_season ds ON ds.season_id = f.season_id
                WHERE f.player_code = ? AND f.minutes > 0 {season_clause}
                GROUP BY ds.season_label, venue
                ORDER BY f.season_id
                """,
                params=params,
            )
            if not df_seas.empty:
                fig = px.bar(df_seas, x="season", y="avg_pts", color="venue",
                             barmode="group", title="Avg pts by season and venue")
                st.plotly_chart(fig, use_container_width=True)
        if any(col in ["xg", "xa"] for col in df.columns):
            xg_seasons_played = query_db(
                "SELECT DISTINCT season_id FROM fact_gw_player WHERE player_code = ? AND expected_goals IS NOT NULL",
                params=(player_code,),
            )
            if xg_seasons_played.empty:
                st.caption("xG/xA are NULL for pre-2022-23 seasons.")

# ---- Template 16: Attacking Returns ------------------------------------------
elif template.startswith("16"):
    st.markdown("**Attacking Returns** — Goals + assists leaders.")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t16_season")
        season_id   = season_map[season_lbl]
    with c2:
        sort_by     = st.selectbox("Sort by",
                                   ["Goal involvements", "Goals", "Assists", "xGI (xG era)"],
                                   key="t16_sort")
    with c3:
        pos_sel     = st.selectbox("Position", pos_options, key="t16_pos")
    with c4:
        min_apps    = st.number_input("Min appearances", min_value=1, max_value=38, value=5, key="t16_apps")
    with c5:
        top_n       = st.number_input("Top N", min_value=1, max_value=200, value=20, key="t16_n")

    sort_col_map = {
        "Goals": "goals",
        "Assists": "assists",
        "Goal involvements": "goal_involvements",
        "xGI (xG era)": "xgi",
    }

    if st.button("Run", key="t16_run"):
        _xg_warning(season_id)
        sc = sort_col_map[sort_by]
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        params     = (season_id, pos_sel) if pos_sel != "All" else (season_id,)
        df = query_db(
            f"""
            SELECT
                dp.web_name                                     AS player,
                dt.team_name                                    AS team,
                dps.position_label                              AS position,
                SUM(f.goals_scored)                             AS goals,
                SUM(f.assists)                                  AS assists,
                SUM(f.goals_scored) + SUM(f.assists)            AS goal_involvements,
                ROUND(SUM(f.expected_goals), 2)                 AS xg,
                ROUND(SUM(f.expected_assists), 2)               AS xa,
                ROUND(SUM(f.expected_goal_involvements), 2)     AS xgi,
                COUNT(DISTINCT f.fixture_id)                    AS appearances,
                dps.total_points                                AS season_pts
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ? AND f.minutes > 0 {pos_clause}
            GROUP BY f.player_code
            HAVING appearances >= {int(min_apps)}
            ORDER BY {sc} DESC
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        _result_block(df, f"attacking_returns_{season_lbl}.csv")

# ---- Template 17: Double Gameweek Finder -------------------------------------
elif template.startswith("17"):
    st.markdown("**Double Gameweek Finder** — Players with two fixtures in one GW.")
    c1, c2, c3 = st.columns(3)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t17_season")
        season_id   = season_map[season_lbl]

    # Auto-detect DGW gameweeks
    dgw_df = query_db(
        """
        SELECT DISTINCT gw FROM fact_gw_player
        WHERE season_id = ?
        GROUP BY player_code, gw
        HAVING COUNT(DISTINCT fixture_id) > 1
        ORDER BY gw
        """,
        params=(season_id,),
    )

    with c2:
        if dgw_df.empty:
            st.info(f"No double gameweeks detected in {season_lbl}.")
            gw_num = None
        else:
            gw_opts = dgw_df["gw"].tolist()
            gw_num  = st.selectbox("DGW Gameweek", gw_opts, key="t17_gw")
    with c3:
        pos_sel = st.selectbox("Position", pos_options, key="t17_pos")

    if gw_num and st.button("Run", key="t17_run"):
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        params     = (season_id, int(gw_num), pos_sel) if pos_sel != "All" else (season_id, int(gw_num))
        df = query_db(
            f"""
            SELECT
                dp.web_name                         AS player,
                dt.team_name                        AS team,
                dps.position_label                  AS position,
                COUNT(DISTINCT f.fixture_id)        AS fixtures,
                SUM(f.total_points)                 AS total_pts,
                SUM(f.minutes)                      AS total_minutes,
                SUM(f.goals_scored)                 AS goals,
                SUM(f.assists)                      AS assists,
                ROUND(MAX(f.value) / 10.0, 1)      AS price_m
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ? AND f.gw = ? {pos_clause}
            GROUP BY f.player_code
            HAVING fixtures > 1
            ORDER BY total_pts DESC
            """,
            params=params,
        )
        _result_block(df, f"dgw_gw{int(gw_num)}_{season_lbl}.csv")

# ---- Template 18: GK-Specific Stats ------------------------------------------
elif template.startswith("18"):
    st.markdown("**GK-Specific Stats** — Saves, clean sheets, penalty saves.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t18_season")
        season_id   = season_map[season_lbl]
    with c2:
        sort_by     = st.selectbox("Sort by",
                                   ["Total FPL pts", "Total saves", "Clean sheets", "Save pts"],
                                   key="t18_sort")
    with c3:
        min_apps    = st.number_input("Min appearances", min_value=1, max_value=38, value=10, key="t18_apps")
    with c4:
        top_n       = st.number_input("Top N", min_value=1, max_value=50, value=20, key="t18_n")

    sort_col_map = {
        "Total FPL pts":  "season_pts",
        "Total saves":    "total_saves",
        "Clean sheets":   "clean_sheets",
        "Save pts":       "save_pts",
    }

    if st.button("Run", key="t18_run"):
        sc = sort_col_map[sort_by]
        df = query_db(
            f"""
            SELECT
                dp.web_name                                     AS player,
                dt.team_name                                    AS team,
                COUNT(DISTINCT f.fixture_id)                    AS appearances,
                SUM(f.saves)                                    AS total_saves,
                ROUND(AVG(f.saves), 1)                          AS avg_saves_per_gw,
                ROUND(SUM(f.saves) / 3.0, 0)                   AS save_pts,
                SUM(f.clean_sheets)                             AS clean_sheets,
                ROUND(100.0 * SUM(f.clean_sheets)
                      / COUNT(DISTINCT f.fixture_id), 1)        AS cs_rate_pct,
                SUM(f.penalties_saved)                          AS pen_saves,
                SUM(f.bonus)                                    AS total_bonus,
                dps.total_points                                AS season_pts,
                ROUND(dps.start_cost / 10.0, 1)                AS start_price_m
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ?
              AND dps.position_label = 'GK'
              AND f.minutes >= 60
            GROUP BY f.player_code
            HAVING appearances >= {int(min_apps)}
            ORDER BY {sc} DESC
            LIMIT {int(top_n)}
            """,
            params=(season_id,),
        )
        _result_block(df, f"gk_stats_{season_lbl}.csv")
        st.caption(
            "`goals_conceded` shown here is time-on-pitch scoped (per player, not team total). "
            "See Known Data Quirks on the Bias & Quality page."
        )

# ---- Template 19: Suspension Risk --------------------------------------------
elif template.startswith("19"):
    st.markdown("**Suspension Risk** — Yellow/red card accumulation.")
    c1, c2, c3 = st.columns(3)
    with c1:
        season_lbl  = st.selectbox("Season", season_labels, key="t19_season")
        season_id   = season_map[season_lbl]
    with c2:
        pos_sel     = st.selectbox("Position", pos_options, key="t19_pos")
    with c3:
        top_n       = st.number_input("Top N", min_value=1, max_value=200, value=30, key="t19_n")

    if st.button("Run", key="t19_run"):
        pos_clause = "" if pos_sel == "All" else "AND dps.position_label = ?"
        params     = (season_id, pos_sel) if pos_sel != "All" else (season_id,)
        df = query_db(
            f"""
            SELECT
                dp.web_name                             AS player,
                dt.team_name                            AS team,
                dps.position_label                      AS position,
                SUM(f.yellow_cards)                     AS yellow_cards,
                SUM(f.red_cards)                        AS red_cards,
                COUNT(DISTINCT f.fixture_id)            AS appearances,
                ROUND(100.0 * SUM(f.yellow_cards)
                      / COUNT(DISTINCT f.fixture_id), 1) AS yellow_per_100_gws,
                dps.total_points                        AS season_pts
            FROM fact_gw_player f
            JOIN dim_player        dp  ON dp.player_code  = f.player_code
            JOIN dim_player_season dps ON dps.player_code = f.player_code
                                      AND dps.season_id  = f.season_id
            JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
            WHERE f.season_id = ? AND f.minutes > 0 {pos_clause}
            GROUP BY f.player_code
            ORDER BY yellow_cards DESC, red_cards DESC
            LIMIT {int(top_n)}
            """,
            params=params,
        )
        if not df.empty:
            df["at_risk"] = df["yellow_cards"] >= 5
        _result_block(df, f"suspension_risk_{season_lbl}.csv")
        st.caption("Players with 5+ yellow cards are typically at risk of a one-match ban.")

# ---- Template 20: Season History Explorer ------------------------------------
elif template.startswith("20"):
    st.markdown(
        "**Season History Explorer** — Full FPL history via `fact_player_season_history`. "
        "Covers seasons outside the main DB scope."
    )
    player_code = _player_search("Player name", "t20_player")

    if player_code and st.button("Run", key="t20_run"):
        df = query_db(
            """
            SELECT
                ds.season_label                             AS season,
                h.total_points,
                h.minutes,
                h.goals_scored,
                h.assists,
                h.clean_sheets,
                h.saves,
                h.bonus,
                ROUND(h.start_cost / 10.0, 1)             AS start_price_m,
                ROUND(h.end_cost   / 10.0, 1)             AS end_price_m,
                ROUND((h.end_cost - h.start_cost) / 10.0, 1) AS price_change_m
            FROM fact_player_season_history h
            JOIN dim_season ds ON ds.season_id = h.season_id
            WHERE h.player_code = ?
            ORDER BY h.season_id
            """,
            params=(player_code,),
        )
        _result_block(df, "season_history.csv")
        if not df.empty:
            fig = px.bar(df, x="season", y="total_points",
                         title="Career FPL points (all available seasons)",
                         labels={"total_points": "FPL pts"})
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section B — Table Browser
# ---------------------------------------------------------------------------

st.subheader("Table Browser")

with st.expander("Browse raw tables"):
    c1, c2 = st.columns([2, 1])
    with c1:
        table_sel = st.selectbox("Table", DB_TABLES, key="tb_table")
    with c2:
        row_limit = st.number_input("Rows to show", min_value=1, max_value=500, value=50, key="tb_rows")

    # Dynamic column filters
    col_df = query_db(f"SELECT * FROM {table_sel} LIMIT 0")
    filters: dict[str, str] = {}
    if not col_df.empty or True:
        sample = query_db(f"SELECT * FROM {table_sel} LIMIT 1")
        if not sample.empty:
            filter_cols = st.multiselect(
                "Filter by columns (optional)",
                list(sample.columns),
                key="tb_filter_cols",
            )
            if filter_cols:
                fcols = st.columns(len(filter_cols))
                for col_widget, col_name in zip(fcols, filter_cols):
                    val = col_widget.text_input(f"{col_name}", key=f"tb_f_{col_name}")
                    if val:
                        filters[col_name] = val

    if st.button("Browse", key="tb_run"):
        if filters:
            where_parts = [f"CAST({col} AS TEXT) LIKE ?" for col in filters]
            where_clause = "WHERE " + " AND ".join(where_parts)
            params = tuple(f"%{v}%" for v in filters.values())
        else:
            where_clause = ""
            params = ()
        df = query_db(
            f"SELECT * FROM {table_sel} {where_clause} LIMIT {int(row_limit)}",
            params=params,
        )
        _result_block(df, f"{table_sel}_browse.csv")

st.divider()

# ---------------------------------------------------------------------------
# Section C — Free-Form SQL
# ---------------------------------------------------------------------------

st.subheader("Free-Form SQL")
st.caption(
    "Read-only. Write any SELECT query against the FPL database. "
    "The connection is opened in `?mode=ro` — no writes are possible."
)

sql_input = st.text_area(
    "SQL query",
    height=150,
    placeholder="SELECT * FROM dim_player LIMIT 10",
    key="freeform_sql",
)

if st.button("Run SQL", key="freeform_run"):
    if not sql_input.strip():
        st.warning("Enter a SQL query first.")
    else:
        try:
            result = query_db(sql_input.strip())
            if result.empty:
                st.info("Query returned no rows.")
            else:
                st.dataframe(result, use_container_width=True)
                st.caption(f"{len(result):,} rows returned")
                st.download_button(
                    "Download CSV",
                    result.to_csv(index=False).encode(),
                    file_name="query_result.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Query error: {e}")
