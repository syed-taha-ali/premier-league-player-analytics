"""
Page 4 — GW Predictions

Primary weekly decision tool: ranked player predictions with FDR, captain candidates,
ownership bubble chart, and CSV download.
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
    list_available_gws,
    load_fdr_calendar,
    load_predictions,
    load_team_names,
)

st.set_page_config(layout="wide", page_title="FPL Analysis — GW Predictions")

st.title("GW Predictions")
st.caption("Ranked player predictions. Use the sidebar to filter by GW, model, and position.")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

available_gws = list_available_gws()

if not available_gws:
    st.warning(
        "No prediction files found. Run `python run_gw.py` to generate predictions."
    )
    st.stop()

with st.sidebar:
    st.header("Filters")

    gw_labels = [f"GW {gw} — Season {sid}" for gw, sid in available_gws]
    gw_idx    = st.selectbox("Gameweek", options=range(len(available_gws)), format_func=lambda i: gw_labels[i])
    sel_gw, sel_season = available_gws[gw_idx]

    df_raw = load_predictions(sel_gw, sel_season)
    if df_raw.empty:
        st.stop()

    # Model selector — only show models that exist in this CSV
    pred_cols_available = [c for c in ["pred_ridge", "pred_bayesian_ridge", "pred_blending", "pred_ensemble"]
                           if c in df_raw.columns]
    model_display = {
        "pred_ridge":         "Ridge",
        "pred_bayesian_ridge":"Bayesian Ridge",
        "pred_blending":      "Blending",
        "pred_ensemble":      "Ensemble",
    }
    model_col = st.selectbox(
        "Model",
        options=pred_cols_available,
        format_func=lambda c: model_display.get(c, c),
    )

    positions_sel = st.multiselect("Position", POSITIONS, default=POSITIONS)

    price_min, price_max = 4.0, 15.0
    if "value_lag1" in df_raw.columns:
        prices = df_raw["value_lag1"].dropna() / 10
        price_min = float(prices.min())
        price_max = float(prices.max())
    price_range = st.slider(
        "Price band (£m)",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        step=0.1,
    )

    top_n = st.number_input("Top-N per position", min_value=1, max_value=100, value=20)
    show_differentials = st.checkbox("Show differentials only", value=False)

# ---------------------------------------------------------------------------
# Apply filters and derived columns
# ---------------------------------------------------------------------------

df = df_raw.copy()

# Price in £m
if "value_lag1" in df.columns:
    df["price_m"] = (df["value_lag1"] / 10).round(1)
else:
    df["price_m"] = np.nan

# FDR from opponent_season_rank: rank 1–6 = FDR 5 (hardest), 18–20 = FDR 1 (easiest)
def _fdr(rank: float | None) -> float:
    """Map end-of-season opponent rank (1=strongest) to FDR score (5=hardest fixture)."""
    if pd.isna(rank):
        return np.nan
    r = float(rank)
    if r <= 6:   return 5
    if r <= 10:  return 4
    if r <= 14:  return 3
    if r <= 17:  return 2
    return 1

if "opponent_season_rank" in df.columns:
    df["fdr"] = df["opponent_season_rank"].apply(_fdr)
else:
    df["fdr"] = np.nan

# Differential: predicted >= position median AND ownership < 10%
if model_col in df.columns:
    pos_medians   = df.groupby("position")[model_col].median()
    df["pos_med"] = df["position"].map(pos_medians)
    own_col_avail = "ownership_pct" in df.columns
    if own_col_avail:
        df["differential"] = (
            (df[model_col] >= df["pos_med"]) & (df["ownership_pct"] < 10)
        )
    else:
        df["differential"] = False
else:
    df["differential"] = False

# Filter by position
df = df[df["position"].isin(positions_sel)]

# Filter by price
if "price_m" in df.columns:
    df = df[
        (df["price_m"] >= price_range[0]) & (df["price_m"] <= price_range[1])
    ]

# Differentials filter
if show_differentials:
    df = df[df["differential"]]

# Top-N per position
if model_col in df.columns:
    df = (
        df.sort_values(model_col, ascending=False)
        .groupby("position", group_keys=False)
        .head(int(top_n))
    )

# ---------------------------------------------------------------------------
# Section A — Fixture Difficulty Calendar
# ---------------------------------------------------------------------------

with st.expander("Fixture Difficulty Calendar", expanded=False):
    fdr_df = load_fdr_calendar(sel_season)
    if fdr_df.empty:
        st.info("FDR calendar unavailable — feature matrix not found.")
    else:
        fdr_gws = sorted(fdr_df["gw"].unique())
        teams   = sorted(fdr_df["team_name"].unique())

        # Build pivot: team x GW -> fdr
        pivot = fdr_df.pivot_table(
            index="team_name", columns="gw", values="fdr", aggfunc="first"
        ).reindex(index=teams, columns=fdr_gws)

        # Opponent abbreviations as annotation text
        abbrev_pivot = fdr_df.pivot_table(
            index="team_name", columns="gw",
            values="opponent_team_name", aggfunc="first"
        ).reindex(index=teams, columns=fdr_gws)
        abbrev_text = abbrev_pivot.map(
            lambda x: x[:3].upper() if isinstance(x, str) else ""
        ).values

        fig_cal = px.imshow(
            pivot,
            color_continuous_scale=["#00d26a", "#7fcc7f", "#ffd700", "#ff8c00", "#d9534f"],
            zmin=1, zmax=5,
            aspect="auto",
            labels={"color": "FDR (1=easy, 5=hard)"},
            title=f"Fixture Difficulty — Season {sel_season}",
        )
        fig_cal.update_traces(text=abbrev_text, texttemplate="%{text}", textfont_size=9)
        fig_cal.update_layout(height=600, coloraxis_showscale=True)
        st.plotly_chart(fig_cal, use_container_width=True)
        st.caption(
            "FDR 1 (green) = easy fixture, FDR 5 (red) = hard fixture. "
            "Cell text = opponent abbreviation. "
            "Future fixture data requires a live FPL API call via `etl/fetch.py`."
        )

st.divider()

# ---------------------------------------------------------------------------
# Captain Candidates card
# ---------------------------------------------------------------------------

st.subheader("Captain Candidates")

if model_col in df.columns and not df.empty:
    name_col = "web_name" if "web_name" in df.columns else "player_code"
    top3 = df_raw.copy()  # use unfiltered for captain candidates
    if "position" in top3.columns:
        top3 = top3[top3["position"].isin(positions_sel)]
    top3 = top3.nlargest(3, model_col)

    cols = st.columns(3)
    for col, (_, row) in zip(cols, top3.iterrows()):
        name  = row.get("web_name", str(row.get("player_code", "?")))
        pos   = row.get("position", "")
        team  = row.get("team", "")
        pts   = row[model_col]
        opp   = row.get("opponent_team", "")
        ha    = row.get("home_away", "")
        fdr_v = row.get("fdr", "")

        delta_parts = [team]
        if opp:
            delta_parts.append(f"vs {opp} ({ha})")
        if fdr_v and not pd.isna(fdr_v):
            delta_parts.append(f"FDR {int(fdr_v)}")

        col.metric(
            label=f"{name} ({pos})",
            value=f"{pts:.1f} pts",
            delta=" | ".join(delta_parts),
            delta_color="off",
        )
else:
    st.info("No predictions available for captain selection with current filters.")

st.divider()

# ---------------------------------------------------------------------------
# Main predictions table
# ---------------------------------------------------------------------------

st.subheader(f"GW {sel_gw} — Season {sel_season} Predictions")

if df.empty:
    st.warning("No rows match the current filters.")
else:
    name_col = "web_name" if "web_name" in df.columns else "player_code"

    # Build display columns in order
    display_cols  = [name_col, "position"]
    col_cfg: dict = {}

    if "team" in df.columns:
        display_cols.append("team")

    if "opponent_team" in df.columns and "home_away" in df.columns:
        df["opponent_display"] = df["opponent_team"].fillna("") + " (" + df["home_away"].fillna("") + ")"
        display_cols.append("opponent_display")
        col_cfg["opponent_display"] = st.column_config.TextColumn("Opponent")

    if "fdr" in df.columns:
        display_cols.append("fdr")
        col_cfg["fdr"] = st.column_config.NumberColumn(
            "FDR",
            help="1=easy (green) → 5=hard (red)",
            min_value=1, max_value=5,
            format="%d",
        )

    if "price_m" in df.columns:
        display_cols.append("price_m")
        col_cfg["price_m"] = st.column_config.NumberColumn("Price (£m)", format="%.1f")

    if model_col in df.columns:
        display_cols.append(model_col)
        col_cfg[model_col] = st.column_config.NumberColumn(
            f"Pred pts ({model_display.get(model_col, model_col)})", format="%.2f"
        )

    # Uncertainty band (bayesian_ridge only)
    if model_col == "pred_bayesian_ridge" and "pred_bayesian_ridge_std" in df.columns:
        display_cols.append("pred_bayesian_ridge_std")
        col_cfg["pred_bayesian_ridge_std"] = st.column_config.NumberColumn(
            "Uncertainty (±)", format="%.2f"
        )

    if "ownership_pct" in df.columns:
        display_cols.append("ownership_pct")
        col_cfg["ownership_pct"] = st.column_config.NumberColumn(
            "Ownership %", format="%.1f%%"
        )

    if "differential" in df.columns:
        display_cols.append("differential")
        col_cfg["differential"] = st.column_config.CheckboxColumn(
            "Differential",
            help="High predicted pts + low ownership (<10%)",
        )

    if "total_points" in df.columns:
        df["actual_pts"] = df["total_points"].where(df["total_points"].notna(), other=None)
        display_cols.append("actual_pts")
        col_cfg["actual_pts"] = st.column_config.NumberColumn("Actual pts", format="%.0f")

    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[display_cols].reset_index(drop=True),
        use_container_width=True,
        column_config=col_cfg,
    )

    if "total_points" not in df.columns or df["total_points"].isna().all():
        st.caption("Actual pts will populate after GW results are published.")

    # Download
    csv_bytes = df[display_cols].to_csv(index=False).encode()
    st.download_button(
        label="Download filtered predictions (CSV)",
        data=csv_bytes,
        file_name=f"gw{sel_gw}_s{sel_season}_predictions_filtered.csv",
        mime="text/csv",
    )

st.divider()

# ---------------------------------------------------------------------------
# Section C — Ownership vs Predicted Pts bubble chart
# ---------------------------------------------------------------------------

st.subheader("Ownership vs Predicted Points")

if model_col not in df.columns or df.empty:
    st.info("No data available for bubble chart.")
elif "ownership_pct" not in df.columns:
    st.info("Ownership data unavailable for this GW.")
else:
    df_bubble = df.dropna(subset=[model_col, "ownership_pct"])
    if df_bubble.empty:
        st.info("No ownership data found for selected players.")
    else:
        med_own = df_bubble["ownership_pct"].median()
        med_pts = df_bubble[model_col].median()

        name_col = "web_name" if "web_name" in df_bubble.columns else "player_code"
        hover_cols = [name_col, "position"]
        if "team" in df_bubble.columns:
            hover_cols.append("team")
        if "opponent_team" in df_bubble.columns:
            hover_cols.append("opponent_team")
        if "fdr" in df_bubble.columns:
            hover_cols.append("fdr")

        size_col = "price_m" if "price_m" in df_bubble.columns else None

        fig_bub = px.scatter(
            df_bubble,
            x="ownership_pct",
            y=model_col,
            size=size_col if size_col else None,
            color="position",
            hover_data=hover_cols,
            labels={
                "ownership_pct": "Ownership %",
                model_col: f"Predicted pts ({model_display.get(model_col, model_col)})",
            },
            title="Ownership vs Predicted Points",
        )

        # Quadrant reference lines
        fig_bub.add_hline(y=med_pts, line_dash="dot", line_color="grey", line_width=1)
        fig_bub.add_vline(x=med_own, line_dash="dot", line_color="grey", line_width=1)

        # Quadrant annotations
        x_range = df_bubble["ownership_pct"].max() - df_bubble["ownership_pct"].min()
        y_range = df_bubble[model_col].max() - df_bubble[model_col].min()
        ax_min  = df_bubble["ownership_pct"].min()
        ax_max  = df_bubble["ownership_pct"].max()
        ay_min  = df_bubble[model_col].min()
        ay_max  = df_bubble[model_col].max()

        annotations = [
            dict(x=ax_min + x_range * 0.05, y=ay_max - y_range * 0.05,
                 text="Differentials", showarrow=False, font=dict(size=11, color="#555")),
            dict(x=ax_max - x_range * 0.05, y=ay_max - y_range * 0.05,
                 text="Template picks", showarrow=False, font=dict(size=11, color="#555")),
            dict(x=ax_min + x_range * 0.05, y=ay_min + y_range * 0.05,
                 text="Avoid", showarrow=False, font=dict(size=11, color="#555")),
            dict(x=ax_max - x_range * 0.05, y=ay_min + y_range * 0.05,
                 text="Ownership trap", showarrow=False, font=dict(size=11, color="#555")),
        ]
        fig_bub.update_layout(annotations=annotations, height=500)
        st.plotly_chart(fig_bub, use_container_width=True)
