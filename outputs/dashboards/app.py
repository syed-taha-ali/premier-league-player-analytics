"""
FPL Analysis Dashboard — Landing Page

Shows the latest GW monitoring summary and navigation guide.
Run from the project root:
    streamlit run outputs/dashboards/app.py
"""

import streamlit as st

from utils import (
    list_available_gws,
    load_monitoring_log,
    load_predictions,
    POSITIONS,
)

st.set_page_config(
    layout="wide",
    page_title="FPL Analysis Dashboard",
    page_icon=":soccer:",
)

st.title("FPL Analysis Dashboard")
st.caption(
    "End-to-end Fantasy Premier League analytics pipeline — "
    "Phases 1-9 complete. Use the sidebar to navigate between pages."
)

st.divider()

# ---------------------------------------------------------------------------
# Monitoring summary: latest GW MAE cards
# ---------------------------------------------------------------------------

st.subheader("Latest GW Monitoring Summary")

log = load_monitoring_log()

if log.empty:
    st.info("No monitoring data yet. Run run_gw.py after the first GW.")
else:
    # Get the most recent GW/season combination
    latest = log.sort_values(["season_id", "gw"], ascending=False).iloc[0]
    latest_gw     = int(latest["gw"])
    latest_season = int(latest["season_id"])

    st.markdown(f"**GW {latest_gw} — Season {latest_season}**")

    gw_log = log[(log["gw"] == latest_gw) & (log["season_id"] == latest_season)]

    cols = st.columns(len(POSITIONS))
    for col, pos in zip(cols, POSITIONS):
        row = gw_log[gw_log["position"] == pos]
        if row.empty:
            col.metric(label=pos, value="—")
            continue

        row = row.iloc[0]
        mae       = float(row["mae"])
        threshold = float(row["threshold"])
        alert     = bool(row["alert"])
        status    = "ALERT" if alert else "PASS"
        delta_str = f"Threshold: {threshold:.3f}"

        col.metric(
            label=f"{pos} — {status}",
            value=f"MAE {mae:.3f}",
            delta=delta_str,
            delta_color="inverse" if alert else "off",
        )

    # Rolling trend mini-table
    with st.expander("Full monitoring log (latest 20 rows)"):
        display_cols = ["season_id", "gw", "position", "model", "mae", "rmse",
                        "spearman", "rolling_mae_5gw", "alert"]
        display_cols = [c for c in display_cols if c in log.columns]
        st.dataframe(
            log.sort_values(["season_id", "gw"], ascending=False)
               .head(20)[display_cols]
               .reset_index(drop=True),
            use_container_width=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Latest predictions quick-look
# ---------------------------------------------------------------------------

st.subheader("Latest GW Predictions")

available_gws = list_available_gws()
if not available_gws:
    st.warning(
        "No prediction files found. Run run_gw.py to generate predictions."
    )
else:
    latest_gw_pred, latest_season_pred = available_gws[0]
    st.markdown(f"**GW {latest_gw_pred} — Season {latest_season_pred}** (top 5 per position)")

    df_pred = load_predictions(latest_gw_pred, latest_season_pred)
    if not df_pred.empty:
        # Pick best available prediction column to rank by
        rank_col = next(
            (c for c in ["pred_ridge", "pred_blending", "pred_ensemble"]
             if c in df_pred.columns),
            None,
        )
        name_col  = "web_name" if "web_name" in df_pred.columns else "player_code"
        show_cols = [name_col, "position"]
        if "opponent_team" in df_pred.columns:
            show_cols.append("opponent_team")
        if "home_away" in df_pred.columns:
            show_cols.append("home_away")
        if rank_col:
            show_cols.append(rank_col)
        if "total_points" in df_pred.columns:
            show_cols.append("total_points")

        tabs = st.tabs(POSITIONS)
        for tab, pos in zip(tabs, POSITIONS):
            with tab:
                subset = df_pred[df_pred["position"] == pos]
                if rank_col:
                    subset = subset.sort_values(rank_col, ascending=False)
                st.dataframe(
                    subset.head(5)[show_cols].reset_index(drop=True),
                    use_container_width=True,
                )

st.divider()

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------

st.subheader("Navigation Guide")

nav_data = {
    "Page": [
        "1 — Data Explorer",
        "2 — Bias & Quality",
        "3 — Model Performance",
        "4 — GW Predictions",
        "5 — Player Scouting",
        "6 — Database Explorer",
    ],
    "Purpose": [
        "Historical scoring distributions, home/away splits, team strength, career trajectories",
        "Known data biases, schema eras, missing data, known quirks",
        "CV metrics, OOF calibration, SHAP plots, residual decomposition, monitoring trends",
        "Ranked GW prediction table with FDR, captain candidates, ownership bubble chart",
        "Boom/bust quadrant, value picks, player comparison, price trajectory, component model",
        "20 preset query templates (Player / Team / Gameweek / Advanced), table browser, free-form SQL",
    ],
}

st.table(nav_data)
