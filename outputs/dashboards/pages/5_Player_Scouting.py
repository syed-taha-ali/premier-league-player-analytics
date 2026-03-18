"""
Page 5 — Player Scouting

Boom/bust quadrant, value picks, form vs price, player comparison,
price trajectory, and component model OOF analysis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    POSITIONS,
    TRAIN_DIR,
    list_available_gws,
    load_oof,
    load_player_names,
    load_predictions,
    load_season_list,
    query_db,
)

st.set_page_config(layout="wide", page_title="FPL Analysis — Player Scouting")

st.title("Player Scouting")
st.caption(
    "Boom/bust profiling, value picks, player comparison, "
    "price trajectory, and component model analysis."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player_search(label: str, key: str) -> int | None:
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
    opts = dict(zip(matches["web_name"], matches["player_code"]))
    chosen = st.selectbox("Select player", list(opts.keys()), key=f"{key}_sel")
    return opts[chosen]


def _season_options() -> tuple[list[dict], dict]:
    seasons = load_season_list()  # newest first
    sid_to_label = {s["season_id"]: s["season_label"] for s in seasons}
    return seasons, sid_to_label


seasons_list, sid_to_label = _season_options()
latest_sid = seasons_list[0]["season_id"] if seasons_list else 10

# ---------------------------------------------------------------------------
# Section A — Boom/Bust Quadrant
# ---------------------------------------------------------------------------

st.subheader("Boom/Bust Quadrant")
st.caption(
    "Mean vs standard deviation of GW points. "
    "High mean + low SD = reliable banker (captain). "
    "High mean + high SD = boom/bust (great captain, risky bench spot)."
)

c1, c2, c3 = st.columns(3)
with c1:
    bb_season = st.selectbox(
        "Season",
        [s["season_id"] for s in seasons_list],
        format_func=lambda s: sid_to_label.get(s, str(s)),
        key="bb_season",
    )
with c2:
    bb_pos = st.selectbox("Position", ["All"] + POSITIONS, key="bb_pos")
with c3:
    bb_min_apps = st.number_input("Min appearances", min_value=3, max_value=38, value=5, key="bb_apps")

pos_clause = "" if bb_pos == "All" else f"AND dps.position_label = '{bb_pos}'"

bb_raw = query_db(
    f"""
    SELECT
        f.player_code,
        dp.web_name         AS player,
        dps.position_label  AS position,
        dt.team_name        AS team,
        f.total_points,
        ROUND(dps.start_cost / 10.0, 1) AS price_m
    FROM fact_gw_player f
    JOIN dim_player        dp  ON dp.player_code  = f.player_code
    JOIN dim_player_season dps ON dps.player_code = f.player_code
                               AND dps.season_id  = f.season_id
    JOIN dim_team          dt  ON dt.team_sk      = f.team_sk
    WHERE f.season_id = ?
      AND f.minutes > 0
      AND dps.position_label IN ('GK','DEF','MID','FWD')
      {pos_clause}
    """,
    params=(bb_season,),
)

if bb_raw.empty:
    st.info("No data for selected season/position.")
else:
    bb_stats = (
        bb_raw.groupby(["player_code", "player", "position", "team", "price_m"])["total_points"]
        .agg(mean_pts="mean", std_pts="std", appearances="count")
        .reset_index()
    )
    bb_stats = bb_stats[bb_stats["appearances"] >= bb_min_apps].dropna(subset=["std_pts"])
    bb_stats["mean_pts"] = bb_stats["mean_pts"].round(2)
    bb_stats["std_pts"]  = bb_stats["std_pts"].round(2)

    med_mean = bb_stats["mean_pts"].median()
    med_std  = bb_stats["std_pts"].median()

    fig_bb = px.scatter(
        bb_stats,
        x="std_pts",
        y="mean_pts",
        color="position",
        size="price_m",
        size_max=20,
        hover_data=["player", "team", "appearances", "price_m"],
        title=f"Boom/Bust Quadrant — {sid_to_label.get(bb_season, bb_season)}",
        labels={"std_pts": "Std Dev pts (volatility)", "mean_pts": "Mean pts (reliability)"},
        opacity=0.75,
    )
    fig_bb.add_hline(y=med_mean, line_dash="dot", line_color="grey", line_width=1)
    fig_bb.add_vline(x=med_std,  line_dash="dot", line_color="grey", line_width=1)

    # Quadrant annotations
    x_rng = bb_stats["std_pts"]
    y_rng = bb_stats["mean_pts"]
    quads = [
        (x_rng.min(),  y_rng.max(),  "Reliable Banker",   "left",  "top"),
        (x_rng.max(),  y_rng.max(),  "Boom/Bust",         "right", "top"),
        (x_rng.min(),  y_rng.min(),  "Budget Filler",     "left",  "bottom"),
        (x_rng.max(),  y_rng.min(),  "Liability",         "right", "bottom"),
    ]
    for qx, qy, label, xanchor, yanchor in quads:
        fig_bb.add_annotation(
            x=qx, y=qy, text=label, showarrow=False,
            font=dict(size=10, color="#888"),
            xanchor=xanchor, yanchor=yanchor,
        )

    fig_bb.update_layout(height=520)
    st.plotly_chart(fig_bb, use_container_width=True)
    st.caption(f"{len(bb_stats):,} players shown (min {bb_min_apps} appearances).")

st.divider()

# ---------------------------------------------------------------------------
# Section B — Value Picks (Price vs Predicted Points)
# ---------------------------------------------------------------------------

st.subheader("Value Picks")
st.caption("Points-per-million from the latest prediction CSV. Top-3 value picks per position annotated.")

available_gws = list_available_gws()

if not available_gws:
    st.warning("No prediction files found. Run `python run_gw.py` to generate predictions.")
else:
    c1, c2 = st.columns(2)
    with c1:
        gw_labels = [f"GW {gw} — Season {sid}" for gw, sid in available_gws]
        gw_idx    = st.selectbox(
            "Gameweek",
            range(len(available_gws)),
            format_func=lambda i: gw_labels[i],
            key="val_gw",
        )
        sel_gw, sel_sid = available_gws[gw_idx]
    with c2:
        val_pos = st.selectbox("Position filter", ["All"] + POSITIONS, key="val_pos")

    df_pred = load_predictions(sel_gw, sel_sid)

    if not df_pred.empty and "pred_ridge" in df_pred.columns and "value_lag1" in df_pred.columns:
        df_val = df_pred.copy()
        if val_pos != "All":
            df_val = df_val[df_val["position"] == val_pos]
        df_val = df_val[df_val["value_lag1"] > 0].copy()
        df_val["price_m"]       = (df_val["value_lag1"] / 10).round(1)
        df_val["pts_per_m"]     = (df_val["pred_ridge"] / df_val["price_m"]).round(3)
        name_col = "web_name" if "web_name" in df_val.columns else "player_code"

        # Scatter: price vs predicted pts coloured by pts_per_m
        fig_val = px.scatter(
            df_val,
            x="price_m",
            y="pred_ridge",
            color="pts_per_m",
            color_continuous_scale="RdYlGn",
            size="price_m",
            size_max=16,
            hover_data=[name_col, "position", "pts_per_m"] +
                       (["team"] if "team" in df_val.columns else []) +
                       (["opponent_team"] if "opponent_team" in df_val.columns else []),
            title=f"Price vs Predicted Points — GW {sel_gw}",
            labels={"price_m": "Price (£m)", "pred_ridge": "Predicted pts (Ridge)",
                    "pts_per_m": "Pts/£m"},
        )

        # Annotate top-3 value picks per position
        top3_all = (
            df_val.groupby("position", group_keys=False)
            .apply(lambda g: g.nlargest(3, "pts_per_m"))
        )
        for _, row in top3_all.iterrows():
            fig_val.add_annotation(
                x=row["price_m"], y=row["pred_ridge"],
                text=str(row.get(name_col, "")),
                showarrow=True, arrowhead=1, arrowsize=0.8,
                font=dict(size=9), ax=15, ay=-20,
            )

        fig_val.update_layout(height=480)
        st.plotly_chart(fig_val, use_container_width=True)

        # Top-5 value table per position
        st.markdown("**Top 5 value picks per position (pts/£m)**")
        pos_tabs = st.tabs([p for p in POSITIONS if p in df_val["position"].unique()])
        for tab, pos in zip(pos_tabs, [p for p in POSITIONS if p in df_val["position"].unique()]):
            with tab:
                top5 = (
                    df_val[df_val["position"] == pos]
                    .nlargest(5, "pts_per_m")[[name_col, "price_m", "pred_ridge", "pts_per_m"]]
                    .reset_index(drop=True)
                )
                top5.columns = ["Player", "Price (£m)", "Pred pts", "Pts/£m"]
                st.dataframe(top5, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Form vs Price ----
    st.subheader("Form vs Price")
    st.caption("Rolling 5-GW form vs current price. In-form players at their price point.")

    if not df_pred.empty and "pts_rolling_5gw" in df_pred.columns:
        df_form = df_pred.copy()
        if val_pos != "All":
            df_form = df_form[df_form["position"] == val_pos]
        df_form = df_form[df_form["value_lag1"] > 0].copy()
        df_form["price_m"] = (df_form["value_lag1"] / 10).round(1)
        name_col = "web_name" if "web_name" in df_form.columns else "player_code"

        fig_form = px.scatter(
            df_form.dropna(subset=["pts_rolling_5gw"]),
            x="price_m",
            y="pts_rolling_5gw",
            color="position",
            hover_data=[name_col] +
                       (["team"] if "team" in df_form.columns else []) +
                       (["pred_ridge"] if "pred_ridge" in df_form.columns else []),
            title=f"Form vs Price — GW {sel_gw}",
            labels={"price_m": "Price (£m)", "pts_rolling_5gw": "Pts (rolling 5 GW avg)"},
            opacity=0.7,
        )
        fig_form.update_layout(height=430)
        st.plotly_chart(fig_form, use_container_width=True)
    else:
        st.info("pts_rolling_5gw not available in prediction CSV.")

st.divider()

# ---------------------------------------------------------------------------
# Section C — Player Comparison
# ---------------------------------------------------------------------------

st.subheader("Player Comparison")
st.caption("Overlay up to 4 players' rolling form or season trajectory.")

c1, c2, c3 = st.columns(3)
with c1:
    comp_season = st.selectbox(
        "Season",
        [s["season_id"] for s in seasons_list],
        format_func=lambda s: sid_to_label.get(s, str(s)),
        key="comp_season",
    )
with c2:
    comp_metric = st.selectbox(
        "Metric",
        ["pts_rolling_5gw", "pts_rolling_3gw", "season_pts_per_gw"],
        key="comp_metric",
    )
with c3:
    comp_n = st.number_input("Number of players (2–4)", min_value=2, max_value=4, value=2, key="comp_n")

# Player search inputs in columns
search_cols = st.columns(int(comp_n))
player_codes: list[int] = []
player_names_sel: list[str] = []

for i, col in enumerate(search_cols):
    with col:
        q = st.text_input(f"Player {i+1}", key=f"comp_q{i}", placeholder="e.g. Salah")
        if q:
            matches = query_db(
                "SELECT player_code, web_name FROM dim_player WHERE LOWER(web_name) LIKE LOWER(?) LIMIT 20",
                params=(f"%{q}%",),
            )
            if not matches.empty:
                opts   = dict(zip(matches["web_name"], matches["player_code"]))
                chosen = st.selectbox("Select", list(opts.keys()), key=f"comp_sel{i}")
                player_codes.append(opts[chosen])
                player_names_sel.append(chosen)

if len(player_codes) >= 2 and st.button("Compare", key="comp_run"):
    codes_in = ", ".join(str(c) for c in player_codes)

    raw_pts = query_db(
        f"""
        SELECT f.player_code, f.gw, f.season_id, f.total_points, f.minutes
        FROM fact_gw_player f
        WHERE f.season_id = ?
          AND f.player_code IN ({codes_in})
        ORDER BY f.player_code, f.gw
        """,
        params=(comp_season,),
    )

    if raw_pts.empty:
        st.info("No data found for selected players in this season.")
    else:
        # Compute rolling metrics per player
        pname_map = dict(zip(player_codes, player_names_sel))
        frames = []
        for pcode in player_codes:
            pdf = raw_pts[raw_pts["player_code"] == pcode].copy()
            pdf = pdf[pdf["minutes"] > 0].sort_values("gw")
            pdf["pts_rolling_5gw"]       = pdf["total_points"].rolling(5, min_periods=1).mean().round(2)
            pdf["pts_rolling_3gw"]       = pdf["total_points"].rolling(3, min_periods=1).mean().round(2)
            pdf["season_pts_per_gw"]     = pdf["total_points"].expanding().mean().round(2)
            pdf["player_name"]           = pname_map.get(pcode, str(pcode))
            frames.append(pdf)

        comp_df = pd.concat(frames, ignore_index=True)

        fig_comp = px.line(
            comp_df,
            x="gw",
            y=comp_metric,
            color="player_name",
            markers=True,
            title=f"Player Comparison — {comp_metric} — {sid_to_label.get(comp_season, comp_season)}",
            labels={comp_metric: comp_metric.replace("_", " ").title(), "gw": "Gameweek",
                    "player_name": "Player"},
        )
        fig_comp.update_layout(height=440)
        st.plotly_chart(fig_comp, use_container_width=True)

        # Summary table
        summary = (
            comp_df.groupby("player_name")["total_points"]
            .agg(total_pts="sum", appearances="count", avg_pts="mean", best_gw="max")
            .round(2)
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Section D — Price Trajectory
# ---------------------------------------------------------------------------

st.subheader("Price Trajectory")
st.caption("GW-by-GW price movement for a player, with actual points overlay.")

c1, c2 = st.columns(2)
with c1:
    traj_player = _player_search("Player name", "traj_player")
with c2:
    traj_season = st.selectbox(
        "Season",
        [s["season_id"] for s in seasons_list],
        format_func=lambda s: sid_to_label.get(s, str(s)),
        key="traj_season",
    )

if traj_player:
    traj_df = query_db(
        """
        SELECT gw, ROUND(value / 10.0, 1) AS price_m, total_points, minutes
        FROM fact_gw_player
        WHERE player_code = ? AND season_id = ?
        ORDER BY gw
        """,
        params=(traj_player, traj_season),
    )

    if traj_df.empty:
        st.info("No data found for this player in the selected season.")
    else:
        # Per-GW: take max price (handles DGWs) and sum points
        traj_agg = (
            traj_df[traj_df["minutes"] > 0]
            .groupby("gw")
            .agg(price_m=("price_m", "max"), total_points=("total_points", "sum"))
            .reset_index()
        )

        start_price = traj_agg["price_m"].iloc[0]
        end_price   = traj_agg["price_m"].iloc[-1]

        fig_traj = go.Figure()

        # Price line (primary axis)
        fig_traj.add_trace(go.Scatter(
            x=traj_agg["gw"], y=traj_agg["price_m"],
            name="Price (£m)", mode="lines+markers",
            line=dict(color="#636efa", width=2),
            yaxis="y1",
        ))

        # Points bars (secondary axis)
        fig_traj.add_trace(go.Bar(
            x=traj_agg["gw"], y=traj_agg["total_points"],
            name="GW pts", marker_color="rgba(99,202,130,0.5)",
            yaxis="y2",
        ))

        fig_traj.update_layout(
            title=f"Price Trajectory — {sid_to_label.get(traj_season, traj_season)}",
            xaxis=dict(title="Gameweek"),
            yaxis=dict(title="Price (£m)", side="left"),
            yaxis2=dict(title="GW pts", overlaying="y", side="right", showgrid=False),
            legend=dict(x=0.01, y=0.99),
            height=420,
        )
        fig_traj.add_annotation(
            x=traj_agg["gw"].iloc[0],  y=start_price,
            text=f"Start: £{start_price}m", showarrow=True, arrowhead=1,
            font=dict(size=10), ax=30, ay=-25,
        )
        fig_traj.add_annotation(
            x=traj_agg["gw"].iloc[-1], y=end_price,
            text=f"Now: £{end_price}m", showarrow=True, arrowhead=1,
            font=dict(size=10), ax=-30, ay=-25,
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        delta = round(end_price - start_price, 1)
        delta_str = f"+£{delta}m" if delta >= 0 else f"£{delta}m"
        st.metric("Price change this season", f"£{end_price}m", delta=delta_str)

st.divider()

# ---------------------------------------------------------------------------
# Section E — Component Model Sub-predictions (OOF historical)
# ---------------------------------------------------------------------------

st.subheader("Component Model Analysis (OOF)")
st.caption(
    "Historical OOF fold 3 predictions from the component model. "
    "`component_edge` = pred_component_model − pred_ridge "
    "(positive = component model rates higher than Ridge). "
    "`p_starts` = pred_minutes_model / 90 clipped to [0,1] — rotation risk proxy."
)

c1, c2 = st.columns(2)
with c1:
    comp_pos = st.selectbox("Position", POSITIONS, key="cm_pos")
with c2:
    cm_fold = st.selectbox("Fold", [3, 2, 1], key="cm_fold",
                           help="Fold 3 = most recent (seasons 7-9 → season 10 predictions)")

oof_cm = load_oof(comp_pos)

if oof_cm.empty:
    st.info(f"OOF data not available for {comp_pos}.")
elif "pred_component_model" not in oof_cm.columns:
    st.info("pred_component_model not found in OOF parquet.")
else:
    fold_df = oof_cm[oof_cm["fold"] == cm_fold].copy()
    if fold_df.empty:
        st.info(f"No rows for fold {cm_fold}.")
    else:
        fold_df["component_edge"] = (
            fold_df["pred_component_model"] - fold_df["pred_ridge"]
        ).round(3)
        fold_df["p_starts"] = (fold_df["pred_minutes_model"] / 90).clip(0, 1).round(3)
        name_col = "web_name" if "web_name" in fold_df.columns else "player_code"

        # Component edge scatter
        fig_cm = px.scatter(
            fold_df.dropna(subset=["pred_ridge", "pred_component_model"]),
            x="pred_ridge",
            y="pred_component_model",
            color="component_edge",
            color_continuous_scale="RdYlGn",
            hover_data=[name_col, "season_id", "gw", "total_points", "component_edge"],
            opacity=0.55,
            title=f"Component Model vs Ridge — {comp_pos} Fold {cm_fold}",
            labels={"pred_ridge": "Ridge prediction",
                    "pred_component_model": "Component model prediction",
                    "component_edge": "Edge (comp − ridge)"},
        )
        ax_max = max(fold_df["pred_ridge"].max(), fold_df["pred_component_model"].max()) * 1.05
        fig_cm.add_shape(
            type="line", x0=0, y0=0, x1=ax_max, y1=ax_max,
            line=dict(dash="dash", color="grey", width=1),
        )
        fig_cm.update_layout(height=460)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Rotation risk table (low p_starts, high ridge prediction)
        st.markdown("**Rotation Risk: Low P(starts) + High Predicted Pts (top 20)**")
        st.caption("Players the ridge model rates highly but the minutes model thinks may not start.")
        risk_tbl = (
            fold_df[[name_col, "p_starts", "pred_ridge", "pred_component_model",
                      "component_edge", "season_id", "gw"]]
            .dropna(subset=["p_starts", "pred_ridge"])
            .sort_values(["pred_ridge", "p_starts"], ascending=[False, True])
            .head(20)
            .reset_index(drop=True)
        )
        risk_tbl.columns = [c.replace("pred_", "").replace("_", " ").title()
                            for c in risk_tbl.columns]
        st.dataframe(risk_tbl, use_container_width=True, hide_index=True)

        st.info(
            "Live component predictions are available by adding `component_model` and "
            "`minutes_model` to the default model set in `run_gw.py`."
        )
