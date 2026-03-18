"""
Page 1 — Data Explorer

Historical scoring distributions, home/away splits, team strength, player career
trajectories, xG validation scatter, era comparison, and team attack vs defence chart.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import EDA_DIR, POSITIONS, query_db

st.set_page_config(layout="wide", page_title="FPL Analysis — Data Explorer")

st.title("Data Explorer")
st.caption("Interactive EDA: scoring distributions, team strength, player trajectories, and xG validation.")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

XG_ERA_SEASONS = [7, 8, 9, 10]

seasons_df = query_db("SELECT season_id, season_label FROM dim_season ORDER BY season_id")
season_id_to_label = dict(zip(seasons_df["season_id"], seasons_df["season_label"])) \
    if not seasons_df.empty else {}
all_seasons = seasons_df["season_id"].tolist() if not seasons_df.empty else list(range(1, 11))
xg_seasons  = [s for s in all_seasons if s >= 7]

with st.sidebar:
    st.header("Filters")
    season_opts = [(sid, season_id_to_label.get(sid, str(sid))) for sid in all_seasons]
    sel_season_ids = st.multiselect(
        "Season(s)",
        options=[s[0] for s in season_opts],
        format_func=lambda s: season_id_to_label.get(s, str(s)),
        default=xg_seasons,
    )
    if not sel_season_ids:
        sel_season_ids = xg_seasons

    pos_filter = st.selectbox("Position", ["All"] + POSITIONS)

season_in_clause = ", ".join(str(s) for s in sel_season_ids)
season_labels_sel = [season_id_to_label.get(s, str(s)) for s in sel_season_ids]

# ---------------------------------------------------------------------------
# Section A — Points Distribution
# ---------------------------------------------------------------------------

st.subheader("Points Distribution")

_valid_pos = "AND f.position_label IN ('GK','DEF','MID','FWD')"
pos_clause_a = "" if pos_filter == "All" else f"AND f.position_label = '{pos_filter}'"

dist_df = query_db(
    f"""
    SELECT f.season_id, f.position_label AS position, f.total_points
    FROM fact_gw_player f
    WHERE f.total_points > 0
      AND f.season_id IN ({season_in_clause})
      {_valid_pos}
      {pos_clause_a}
    """
)

if dist_df.empty:
    st.info("No data for selected filters.")
else:
    dist_df["season"] = dist_df["season_id"].map(season_id_to_label)
    fig_dist = px.histogram(
        dist_df,
        x="total_points",
        color="position",
        nbins=40,
        facet_col="season",
        facet_col_wrap=2,
        title="GW Points Distribution by Position and Season",
        labels={"total_points": "GW Points", "count": "Fixtures"},
        opacity=0.75,
    )
    fig_dist.update_layout(height=400 * ((len(sel_season_ids) + 1) // 2), bargap=0.05)
    st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section B — Home vs Away Effect
# ---------------------------------------------------------------------------

st.subheader("Home vs Away Effect")
st.caption("Mean GW points by position and venue. Home premium is strongest for defenders.")

ha_df = query_db(
    f"""
    SELECT f.position_label AS position,
           f.was_home,
           AVG(f.total_points) AS mean_pts
    FROM fact_gw_player f
    WHERE f.total_points > 0
      AND f.season_id IN ({season_in_clause})
      {_valid_pos}
      {pos_clause_a}
    GROUP BY f.position_label, f.was_home
    """
)

if ha_df.empty:
    st.info("No home/away data for selected filters.")
else:
    ha_df["venue"] = ha_df["was_home"].map({1: "Home", 0: "Away"})
    ha_df["mean_pts"] = ha_df["mean_pts"].round(3)

    # Sort positions in a consistent order
    pos_order = [p for p in POSITIONS if p in ha_df["position"].unique()]
    ha_df["position"] = pd.Categorical(ha_df["position"], categories=pos_order, ordered=True)
    ha_df = ha_df.sort_values("position")

    fig_ha = px.bar(
        ha_df,
        x="position",
        y="mean_pts",
        color="venue",
        barmode="group",
        title="Mean GW Points: Home vs Away",
        labels={"mean_pts": "Mean pts", "position": "Position", "venue": "Venue"},
        color_discrete_map={"Home": "#00cc96", "Away": "#ef553b"},
    )
    fig_ha.update_layout(height=380)
    st.plotly_chart(fig_ha, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section C — Team Strength Heatmap
# ---------------------------------------------------------------------------

st.subheader("Team Strength Heatmap")
st.caption(
    "Average goals conceded per fixture by team and season. "
    "Derived from match scores (`team_h_score`/`team_a_score`) — not from the player-level "
    "`goals_conceded` column, which is time-on-pitch scoped and unreliable for team totals."
)

heatmap_raw = query_db(
    f"""
    SELECT DISTINCT
        dt.team_name,
        ds.season_label AS season,
        f.fixture_id,
        CASE WHEN f.was_home = 1 THEN f.team_a_score ELSE f.team_h_score END AS goals_conceded_by_team
    FROM fact_gw_player f
    JOIN dim_team   dt ON dt.team_sk   = f.team_sk
    JOIN dim_season ds ON ds.season_id = f.season_id
    WHERE f.season_id IN ({season_in_clause})
    """
)

if heatmap_raw.empty:
    st.info("No team data for selected seasons.")
else:
    heat_agg = (
        heatmap_raw.groupby(["team_name", "season"])["goals_conceded_by_team"]
        .mean()
        .round(2)
        .reset_index()
    )
    pivot = heat_agg.pivot(index="team_name", columns="season", values="goals_conceded_by_team")
    # Sort columns chronologically
    pivot = pivot.reindex(columns=sorted(pivot.columns,
                                          key=lambda s: next((i for i, lbl in season_id_to_label.items()
                                                              if lbl == s), 0)))
    pivot = pivot.sort_values(pivot.columns[-1])  # sort teams by most-recent season

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Avg Goals Conceded per Fixture (lower = better defence)",
        labels={"color": "Avg goals conceded"},
    )
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section D — Player Career Trajectory
# ---------------------------------------------------------------------------

st.subheader("Player Career Trajectory")
st.caption("Search for a player to see their GW points across all available seasons.")

search_q = st.text_input("Player name", placeholder="e.g. Salah", key="career_search")

if search_q:
    matches = query_db(
        "SELECT player_code, web_name FROM dim_player WHERE LOWER(web_name) LIKE LOWER(?) LIMIT 20",
        params=(f"%{search_q}%",),
    )
    if matches.empty:
        st.info(f"No players found matching '{search_q}'.")
    else:
        name_map = dict(zip(matches["web_name"], matches["player_code"]))
        chosen   = st.selectbox("Select player", list(name_map.keys()), key="career_player")
        p_code   = name_map[chosen]

        career_df = query_db(
            """
            SELECT f.gw, f.season_id, f.total_points, f.minutes
            FROM fact_gw_player f
            WHERE f.player_code = ?
            ORDER BY f.season_id, f.gw
            """,
            params=(p_code,),
        )

        if career_df.empty:
            st.info("No GW data found for this player.")
        else:
            career_df["season"] = career_df["season_id"].map(season_id_to_label)

            # Average per GW within each season (handles DGWs)
            career_avg = (
                career_df[career_df["minutes"] > 0]
                .groupby(["season", "gw"], as_index=False)["total_points"]
                .mean()
                .round(2)
            )

            fig_career = px.line(
                career_avg,
                x="gw",
                y="total_points",
                color="season",
                markers=True,
                title=f"{chosen} — GW Points by Season",
                labels={"total_points": "GW pts", "gw": "Gameweek", "season": "Season"},
            )
            # Rolling average overlay (season-level mean as reference)
            for season_lbl in career_avg["season"].unique():
                s_data = career_avg[career_avg["season"] == season_lbl]
                s_mean = s_data["total_points"].mean()
                fig_career.add_hline(
                    y=s_mean, line_dash="dot", line_width=1, opacity=0.4,
                    annotation_text=f"{season_lbl} avg {s_mean:.1f}",
                    annotation_position="bottom right",
                )
            fig_career.update_layout(height=420)
            st.plotly_chart(fig_career, use_container_width=True)

            # Summary table
            season_summary = (
                career_df[career_df["minutes"] > 0]
                .groupby("season")
                .agg(
                    total_pts=("total_points", "sum"),
                    appearances=("total_points", "count"),
                    avg_pts=("total_points", "mean"),
                    best_gw=("total_points", "max"),
                )
                .round(2)
                .reset_index()
            )
            st.dataframe(season_summary, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Section E — xG vs Actual Goals (xG Era)
# ---------------------------------------------------------------------------

st.subheader("xG vs Actual Goals (xG Era)")
st.caption(
    "Does expected goals predict actual goals? Points above the line = overperforming xG. "
    "xG data available from 2022-23 (season 7) onwards."
)

xg_seasons_sel = [s for s in sel_season_ids if s >= 7]
if not xg_seasons_sel:
    st.warning("No xG era seasons selected (seasons 7–10). Select at least one season from 2022-23 onwards.")
else:
    xg_season_in = ", ".join(str(s) for s in xg_seasons_sel)
    xg_pos_clause = "" if pos_filter == "All" else f"AND dps.position_label = '{pos_filter}'"

    xg_df = query_db(
        f"""
        SELECT
            dp.web_name,
            dps.position_label          AS position,
            ds.season_label             AS season,
            ROUND(SUM(f.expected_goals), 2)  AS xg,
            SUM(f.goals_scored)              AS goals,
            COUNT(DISTINCT f.fixture_id)     AS appearances
        FROM fact_gw_player f
        JOIN dim_player        dp  ON dp.player_code  = f.player_code
        JOIN dim_player_season dps ON dps.player_code = f.player_code
                                  AND dps.season_id  = f.season_id
        JOIN dim_season        ds  ON ds.season_id    = f.season_id
        WHERE f.season_id IN ({xg_season_in})
          AND f.minutes > 0
          AND f.expected_goals IS NOT NULL
          {xg_pos_clause}
        GROUP BY f.player_code, f.season_id
        HAVING appearances >= 5
        """
    )

    if xg_df.empty:
        st.info("No xG data for selected filters.")
    else:
        fig_xg = px.scatter(
            xg_df,
            x="xg",
            y="goals",
            color="position",
            hover_data=["web_name", "season", "appearances"],
            opacity=0.65,
            title="Expected Goals vs Actual Goals (xG Era)",
            labels={"xg": "Expected Goals (xG)", "goals": "Actual Goals"},
        )
        ax_max = max(xg_df["xg"].max(), xg_df["goals"].max()) * 1.05
        fig_xg.add_shape(
            type="line", x0=0, y0=0, x1=ax_max, y1=ax_max,
            line=dict(dash="dash", color="grey", width=1),
        )
        fig_xg.add_annotation(
            x=ax_max * 0.75, y=ax_max * 0.85,
            text="x = y (perfect prediction)",
            showarrow=False, font=dict(size=10, color="grey"),
        )
        fig_xg.update_layout(height=480)
        st.plotly_chart(fig_xg, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section F — Era Comparison
# ---------------------------------------------------------------------------

st.subheader("Era Comparison")

era_path = EDA_DIR / "era_comparison.png"
if era_path.exists():
    st.image(str(era_path), use_column_width=True)
    st.caption(
        "Pts/GW by schema era. Pre-xG seasons show a −26.1% pts/GW drift vs xG era — "
        "the primary motivation for restricting the ML pipeline to seasons 7–10 (xG era only). "
        "This era restriction is locked and must not be revisited without strong justification."
    )
else:
    st.info("Era comparison chart not found. Re-run the EDA notebook to regenerate.")

st.divider()

# ---------------------------------------------------------------------------
# Section G — Team Attack vs Defence Strength
# ---------------------------------------------------------------------------

st.subheader("Team Attack vs Defence Strength")
st.caption(
    "Teams top-right (high attack, high defensive weakness) = volatile fixture. "
    "Teams bottom-left = safest for GK/DEF. "
    "Attack = avg goals scored per fixture; Defence = avg goals conceded per fixture."
)

atk_def_raw = query_db(
    f"""
    SELECT DISTINCT
        dt.team_name,
        ds.season_label AS season,
        f.fixture_id,
        CASE WHEN f.was_home = 1 THEN f.team_h_score ELSE f.team_a_score END AS goals_scored_by_team,
        CASE WHEN f.was_home = 1 THEN f.team_a_score ELSE f.team_h_score END AS goals_conceded_by_team
    FROM fact_gw_player f
    JOIN dim_team   dt ON dt.team_sk   = f.team_sk
    JOIN dim_season ds ON ds.season_id = f.season_id
    WHERE f.season_id IN ({season_in_clause})
    """
)

if atk_def_raw.empty:
    st.info("No data for selected seasons.")
else:
    atk_def = (
        atk_def_raw.groupby(["team_name", "season"])
        .agg(
            attack_strength=("goals_scored_by_team", "mean"),
            defensive_weakness=("goals_conceded_by_team", "mean"),
            fixtures=("fixture_id", "nunique"),
        )
        .round(3)
        .reset_index()
    )

    med_atk = atk_def["attack_strength"].median()
    med_def = atk_def["defensive_weakness"].median()

    fig_ad = px.scatter(
        atk_def,
        x="defensive_weakness",
        y="attack_strength",
        color="season",
        text="team_name",
        hover_data=["team_name", "fixtures"],
        title="Team Attack vs Defence Strength",
        labels={
            "defensive_weakness": "Defensive Weakness (avg goals conceded/fixture)",
            "attack_strength":    "Attack Strength (avg goals scored/fixture)",
        },
    )
    fig_ad.update_traces(textposition="top center", textfont_size=9)
    fig_ad.add_hline(y=med_atk, line_dash="dot", line_color="grey", line_width=1,
                     annotation_text="Median attack", annotation_position="bottom right")
    fig_ad.add_vline(x=med_def, line_dash="dot", line_color="grey", line_width=1,
                     annotation_text="Median defence", annotation_position="top left")

    # Quadrant annotations
    x_rng = atk_def["defensive_weakness"]
    y_rng = atk_def["attack_strength"]
    fig_ad.add_annotation(x=x_rng.min(), y=y_rng.max(),
                           text="Safe + Attacking", showarrow=False,
                           font=dict(size=10, color="#555"), xanchor="left")
    fig_ad.add_annotation(x=x_rng.max(), y=y_rng.max(),
                           text="Volatile (great for FWD/MID)", showarrow=False,
                           font=dict(size=10, color="#555"), xanchor="right")
    fig_ad.add_annotation(x=x_rng.min(), y=y_rng.min(),
                           text="Safe fixture for GK/DEF", showarrow=False,
                           font=dict(size=10, color="#555"), xanchor="left")
    fig_ad.add_annotation(x=x_rng.max(), y=y_rng.min(),
                           text="Leaky but low scoring", showarrow=False,
                           font=dict(size=10, color="#555"), xanchor="right")

    fig_ad.update_layout(height=550)
    st.plotly_chart(fig_ad, use_container_width=True)
