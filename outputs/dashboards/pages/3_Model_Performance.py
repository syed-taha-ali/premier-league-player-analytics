"""
Page 3 — Model Performance

CV comparison table, interactive OOF calibration, static diagnostic plots,
residual decomposition, live monitoring trend, and per-GW eval report viewer.
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
    FEATURES_DIR,
    LOG_DIR,
    MODELS_DIR,
    POSITIONS,
    TRAIN_DIR,
    load_cv_metrics,
    load_monitoring_log,
    load_oof,
    query_db,
)

st.set_page_config(layout="wide", page_title="FPL Analysis — Model Performance")

st.title("Model Performance")
st.caption(
    "CV results, OOF calibration, SHAP diagnostics, residual decomposition, "
    "and live monitoring trend."
)

# ---------------------------------------------------------------------------
# Shared: position tabs drive Sections A, C, E
# ---------------------------------------------------------------------------

pos_tabs = st.tabs(POSITIONS)

# ---------------------------------------------------------------------------
# Section A — CV Model Comparison Table
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("CV Model Comparison")
st.caption("Mean metrics across folds. Ridge row highlighted as minimum MAE.")

cv_df = load_cv_metrics()

if cv_df.empty:
    st.warning("CV metrics not found. Run `python -m ml.evaluate` first.")
else:
    for tab, pos in zip(pos_tabs, POSITIONS):
        with tab:
            pos_cv = (
                cv_df[cv_df["position"] == pos]
                .groupby("model")[["mae", "rmse", "spearman", "top10_prec"]]
                .mean()
                .round(3)
                .reset_index()
                .sort_values("mae")
            )
            if pos_cv.empty:
                st.info(f"No CV data for {pos}.")
                continue

            best_mae = pos_cv["mae"].min()

            def _highlight_ridge(row):
                """Highlight Ridge rows green; highlight the best-MAE row yellow."""
                if row["model"] == "ridge":
                    return ["background-color: #d4edda"] * len(row)
                if row["mae"] == best_mae:
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            styled = pos_cv.style.apply(_highlight_ridge, axis=1).format(
                {"mae": "{:.3f}", "rmse": "{:.3f}", "spearman": "{:.3f}", "top10_prec": "{:.3f}"}
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption(
                f"Green = Ridge (production model). Yellow = best MAE if different. "
                f"Baseline gate: models must beat rolling-mean baseline on 2/3 metrics."
            )

st.divider()

# ---------------------------------------------------------------------------
# Section B — Interactive OOF Calibration
# ---------------------------------------------------------------------------

st.subheader("Interactive OOF Calibration")
st.caption(
    "Out-of-fold predictions vs actual points. Hover to identify specific players and GWs. "
    "Points above the dashed line = model under-predicted; below = over-predicted."
)

c1, c2, c3 = st.columns(3)
with c1:
    oof_pos = st.selectbox("Position", POSITIONS, key="oof_pos")
with c2:
    # Discover available model columns from first OOF parquet
    _oof_sample = load_oof(oof_pos)
    pred_model_cols = sorted(
        [c.replace("pred_", "") for c in _oof_sample.columns if c.startswith("pred_")
         and not c.endswith("_std")]
    ) if not _oof_sample.empty else ["ridge"]
    oof_model = st.selectbox("Model", pred_model_cols, index=pred_model_cols.index("ridge")
                             if "ridge" in pred_model_cols else 0, key="oof_model")
with c3:
    oof_fold_opts = ["All", "1", "2", "3"]
    oof_fold = st.selectbox("Fold", oof_fold_opts, key="oof_fold")

oof_df = load_oof(oof_pos)
if oof_df.empty:
    st.info(f"OOF predictions not found for {oof_pos}.")
else:
    pred_col = f"pred_{oof_model}"
    if pred_col not in oof_df.columns:
        st.warning(f"`{pred_col}` not found in OOF parquet for {oof_pos}.")
    else:
        df_cal = oof_df.copy()
        if oof_fold != "All":
            df_cal = df_cal[df_cal["fold"] == int(oof_fold)]

        df_cal = df_cal.dropna(subset=[pred_col, "total_points"])
        df_cal["fold_str"] = df_cal["fold"].astype(str)
        hover_cols = ["web_name", "season_id", "gw", "total_points"] if "web_name" in df_cal.columns \
                     else ["player_code", "season_id", "gw", "total_points"]

        fig_cal = px.scatter(
            df_cal,
            x=pred_col,
            y="total_points",
            color="fold_str",
            hover_data=hover_cols,
            opacity=0.45,
            labels={
                pred_col: f"Predicted pts ({oof_model})",
                "total_points": "Actual pts",
                "fold_str": "Fold",
            },
            title=f"OOF Calibration — {oof_pos} / {oof_model}",
        )
        ax_max = max(df_cal[pred_col].max(), df_cal["total_points"].max()) * 1.05
        fig_cal.add_shape(
            type="line", x0=0, y0=0, x1=ax_max, y1=ax_max,
            line=dict(dash="dash", color="grey", width=1),
        )
        fig_cal.update_layout(height=500)
        st.plotly_chart(fig_cal, use_container_width=True)

        # Summary stats
        corr = df_cal[pred_col].corr(df_cal["total_points"])
        mae  = (df_cal[pred_col] - df_cal["total_points"]).abs().mean()
        c_s, c_m = st.columns(2)
        c_s.metric("Pearson r (pred vs actual)", f"{corr:.3f}")
        c_m.metric("Mean Absolute Error (OOF)", f"{mae:.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Section C — Static Diagnostic Plots
# ---------------------------------------------------------------------------

st.subheader("Static Diagnostic Plots")

for tab, pos in zip(pos_tabs, POSITIONS):
    with tab:
        mae_path   = MODELS_DIR / f"mae_by_fold_{pos}.png"
        shap_path  = MODELS_DIR / f"shap_{pos}.png"
        calib_path = MODELS_DIR / f"calibration_{pos}.png"
        resid_path = MODELS_DIR / f"residuals_{pos}.png"
        lc_path    = MODELS_DIR / "learning_curves.png"

        col1, col2 = st.columns(2)
        with col1:
            if mae_path.exists():
                st.image(str(mae_path), caption=f"MAE by fold — {pos}", use_column_width=True)
            else:
                st.info(f"MAE-by-fold plot not found for {pos}.")
            if calib_path.exists():
                st.image(str(calib_path), caption=f"Calibration — {pos}", use_column_width=True)
        with col2:
            if shap_path.exists():
                st.image(str(shap_path), caption=f"SHAP feature importance — {pos}", use_column_width=True)
            else:
                st.info(f"SHAP plot not found for {pos}.")
            if resid_path.exists():
                st.image(str(resid_path), caption=f"Residuals — {pos}", use_column_width=True)

        if lc_path.exists():
            st.image(str(lc_path), caption="Learning curves (all positions)", use_column_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section D — Live Monitoring Trend
# ---------------------------------------------------------------------------

st.subheader("Live Monitoring Trend")
st.caption("Rolling 5-GW MAE per position. Horizontal lines = alert thresholds (1.5× baseline MAE).")

mon_df = load_monitoring_log()

if mon_df.empty:
    st.info("No monitoring data yet. Run run_gw.py after the first GW.")
else:
    n_gws = mon_df[["season_id", "gw"]].drop_duplicates().shape[0]
    if n_gws < 2:
        st.info(
            f"Monitoring trend will populate as more GWs are run. "
            f"Currently {n_gws} GW(s) logged."
        )

    # Build a composite x-axis label: "S{season} GW{gw}"
    mon_df = mon_df.copy()
    mon_df["gw_label"] = "GW" + mon_df["gw"].astype(str)
    mon_df["sort_key"] = mon_df["season_id"] * 100 + mon_df["gw"]
    mon_df = mon_df.sort_values("sort_key")

    fig_mon = px.line(
        mon_df,
        x="gw_label",
        y="rolling_mae_5gw",
        color="position",
        markers=True,
        title="Rolling 5-GW MAE by Position",
        labels={"rolling_mae_5gw": "Rolling MAE (5 GW)", "gw_label": "Gameweek"},
    )

    # Alert threshold lines per position
    thresholds = mon_df.groupby("position")["threshold"].first()
    colours = {"GK": "#636efa", "DEF": "#ef553b", "MID": "#00cc96", "FWD": "#ab63fa"}
    for pos, thr in thresholds.items():
        fig_mon.add_hline(
            y=thr, line_dash="dot",
            line_color=colours.get(pos, "grey"),
            line_width=1,
            annotation_text=f"{pos} threshold",
            annotation_position="bottom right",
        )

    # Mark alert GWs with red markers
    alerts = mon_df[mon_df["alert"] == 1]
    if not alerts.empty:
        fig_mon.add_trace(go.Scatter(
            x=alerts["gw_label"],
            y=alerts["rolling_mae_5gw"],
            mode="markers",
            marker=dict(symbol="x", size=12, color="red"),
            name="Alert",
            hovertext=alerts["position"] + " — ALERT",
        ))

    fig_mon.update_layout(height=400)
    st.plotly_chart(fig_mon, use_container_width=True)

    with st.expander("Full monitoring log"):
        display_cols = ["season_id", "gw", "position", "model", "mae", "rmse",
                        "spearman", "rolling_mae_5gw", "threshold", "alert", "logged_at"]
        display_cols = [c for c in display_cols if c in mon_df.columns]
        st.dataframe(
            mon_df.sort_values("sort_key", ascending=False)[display_cols].reset_index(drop=True),
            use_container_width=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Section E — Residual Decomposition
# ---------------------------------------------------------------------------

st.subheader("Residual Decomposition")
st.caption(
    "Mean residual (actual − predicted) by feature bucket. "
    "Positive = model underestimates; negative = model overestimates."
)

c1, c2 = st.columns(2)
with c1:
    resid_pos   = st.selectbox("Position", POSITIONS, key="resid_pos")
with c2:
    resid_model_opts = pred_model_cols if not _oof_sample.empty else ["ridge"]
    resid_model = st.selectbox("Model", resid_model_opts,
                               index=resid_model_opts.index("ridge") if "ridge" in resid_model_opts else 0,
                               key="resid_model")

if st.button("Compute residuals", key="resid_run"):
    oof_r = load_oof(resid_pos)
    pred_r = f"pred_{resid_model}"

    if oof_r.empty or pred_r not in oof_r.columns:
        st.warning(f"OOF data or model `{resid_model}` not available for {resid_pos}.")
    else:
        # Join feature matrix for was_home, opponent_season_rank, start_cost
        fm_path = FEATURES_DIR / f"feature_matrix_{resid_pos}.parquet"
        if fm_path.exists():
            fm = pd.read_parquet(fm_path)[
                ["season_id", "gw", "fixture_id", "player_code",
                 "was_home", "opponent_season_rank", "start_cost"]
            ]
            df_r = oof_r[["season_id", "gw", "fixture_id", "player_code",
                           "total_points", "fold", pred_r]].merge(
                fm, on=["season_id", "gw", "fixture_id", "player_code"], how="left"
            )
        else:
            df_r = oof_r[["season_id", "gw", "fixture_id", "player_code",
                           "total_points", "fold", pred_r]].copy()

        # Join minutes from fact_gw_player
        min_df = query_db(
            """
            SELECT season_id, gw, fixture_id, player_code, minutes
            FROM fact_gw_player
            WHERE minutes IS NOT NULL
            """
        )
        if not min_df.empty:
            df_r = df_r.merge(min_df, on=["season_id", "gw", "fixture_id", "player_code"], how="left")

        df_r["residual"] = df_r["total_points"] - df_r[pred_r]
        df_r = df_r.dropna(subset=["residual"])

        bc1, bc2 = st.columns(2)

        # Bucket 1 — Home vs Away
        if "was_home" in df_r.columns:
            df_r["venue"] = df_r["was_home"].map({1: "Home", 0: "Away"})
            grp1 = df_r.groupby("venue")["residual"].mean().reset_index()
            grp1.columns = ["venue", "mean_residual"]
            with bc1:
                fig1 = px.bar(grp1, x="venue", y="mean_residual", color="venue",
                              title="Mean residual: Home vs Away",
                              labels={"mean_residual": "Mean residual (actual − pred)"},
                              color_discrete_map={"Home": "#00cc96", "Away": "#ef553b"})
                fig1.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
                fig1.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig1, use_container_width=True)

        # Bucket 2 — Opponent Rank Tier
        if "opponent_season_rank" in df_r.columns:
            df_r["opp_tier"] = pd.cut(
                df_r["opponent_season_rank"],
                bins=[0, 6, 12, 16, 18, 20],
                labels=["Top 6", "Mid-upper", "Mid-lower", "Bottom 4", "Bottom 2"],
            )
            grp2 = df_r.groupby("opp_tier", observed=True)["residual"].mean().reset_index()
            grp2.columns = ["opp_tier", "mean_residual"]
            with bc2:
                fig2 = px.bar(grp2, x="opp_tier", y="mean_residual",
                              title="Mean residual by opponent tier",
                              labels={"mean_residual": "Mean residual", "opp_tier": "Opponent tier"})
                fig2.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)

        bc3, bc4 = st.columns(2)

        # Bucket 3 — Price Band
        if "start_cost" in df_r.columns:
            df_r["price_band"] = pd.cut(
                df_r["start_cost"] / 10,
                bins=[0, 5, 7, 9, 11, 20],
                labels=["<£5m", "£5-7m", "£7-9m", "£9-11m", ">£11m"],
            )
            grp3 = df_r.groupby("price_band", observed=True)["residual"].mean().reset_index()
            grp3.columns = ["price_band", "mean_residual"]
            with bc3:
                fig3 = px.bar(grp3, x="price_band", y="mean_residual",
                              title="Mean residual by price band",
                              labels={"mean_residual": "Mean residual", "price_band": "Price band"})
                fig3.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
                fig3.update_layout(height=350)
                st.plotly_chart(fig3, use_container_width=True)

        # Bucket 4 — Minutes Bucket
        if "minutes" in df_r.columns:
            df_r["mins_bucket"] = pd.cut(
                df_r["minutes"],
                bins=[0, 45, 60, 75, 90, 200],
                labels=["<45", "45-60", "60-75", "75-90", "90+"],
            )
            grp4 = df_r.groupby("mins_bucket", observed=True)["residual"].mean().reset_index()
            grp4.columns = ["mins_bucket", "mean_residual"]
            with bc4:
                fig4 = px.bar(grp4, x="mins_bucket", y="mean_residual",
                              title="Mean residual by minutes played",
                              labels={"mean_residual": "Mean residual", "mins_bucket": "Minutes bucket"})
                fig4.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
                fig4.update_layout(height=350)
                st.plotly_chart(fig4, use_container_width=True)

        n = len(df_r)
        st.caption(
            f"Based on {n:,} OOF predictions for {resid_pos} / {resid_model}. "
            "Positive residual = model systematically underestimates; negative = overestimates."
        )

st.divider()

# ---------------------------------------------------------------------------
# Section F — Per-GW Narrative Reports
# ---------------------------------------------------------------------------

st.subheader("Per-GW Evaluation Reports")
st.caption("Narrative reports written after each GW run, stored in logs/monitoring/.")

report_files = sorted(LOG_DIR.glob("gw*_s*_eval.md"), reverse=True)

if not report_files:
    st.info("No GW reports yet. Run `python run_gw.py` after the first GW.")
else:
    labels   = [f.stem for f in report_files]
    selected = st.selectbox("Gameweek report", labels)
    content  = (LOG_DIR / f"{selected}.md").read_text(encoding="utf-8")
    st.markdown(content)
