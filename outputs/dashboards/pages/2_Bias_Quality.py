"""
Page 2 — Bias & Data Quality

Known ML biases, feature availability by era, fixture difficulty effect,
price vs performance, and documented data quirks.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import EDA_DIR, FEATURES_DIR, POSITIONS

st.set_page_config(layout="wide", page_title="FPL Analysis — Bias & Data Quality")

st.title("Bias & Data Quality")
st.caption(
    "Known ML biases, schema eras, missing data, and documented data quirks. "
    "These are handled in the ETL and ML pipeline — be aware of them when writing "
    "custom SQL queries or interpreting chart results."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR     = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Section A — Bias Summary
# ---------------------------------------------------------------------------

st.subheader("ML Bias Analysis")

bias_path = DOCS_DIR / "data_biases.md"
if bias_path.exists():
    st.markdown(bias_path.read_text(encoding="utf-8"))
else:
    st.warning(f"data_biases.md not found at `{bias_path}`.")

st.divider()

# ---------------------------------------------------------------------------
# Section B — Feature Availability by Era
# ---------------------------------------------------------------------------

st.subheader("Feature Availability by Era")

missing_path = EDA_DIR / "missing_data_matrix.png"
if missing_path.exists():
    st.image(str(missing_path), use_column_width=True)
    st.caption(
        "Missing data matrix across all seasons and features. "
        "xG/xA/xGI/xGC columns are only available from season 7 (2022-23) onwards — "
        "the primary reason the ML pipeline is scoped to the xG era only."
    )
else:
    st.info("Missing data matrix not found. Re-run the EDA notebook to regenerate.")

st.markdown(
    """
**Schema era summary:**

| Era | Seasons | Key additions | Missing |
|-----|---------|---------------|---------|
| Old Opta | 2016-17 to 2018-19 | Passing, dribble, foul stats | No position/team/xP/xG |
| Stripped | 2019-20 | Core stats only | COVID GW 30–38 gap |
| Modern core | 2020-21, 2021-22 | Position, team, xP | No xG/xA |
| xG era | 2022-23, 2023-24 | xG, xA, xGI, xGC, starts | GW7 absent in 2022-23 |
| Manager era | 2024-25 | `mng_*` columns | Non-manager rows have NULL `mng_*` |
| Defensive era | 2025-26 | defensive_contribution, CBI, recoveries, tackles | Drops `mng_*`, drops `starts` |

The ML pipeline uses xG era seasons (7–10) only. All other seasons are available in the DB for historical queries but excluded from model training.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Section C — Fixture Difficulty Effect
# ---------------------------------------------------------------------------

st.subheader("Fixture Difficulty Effect")

top6_path = EDA_DIR / "top6_fixture_effect.png"
if top6_path.exists():
    st.image(str(top6_path), use_column_width=True)
    st.caption(
        "Top-6 fixture penalty by position: DEF −33.8%, FWD −21.2%, GK −17.6%, MID −16.6%. "
        "`opponent_season_rank` is a mandatory feature in all models."
    )
else:
    st.info("Top-6 fixture effect chart not found. Re-run the EDA notebook to regenerate.")

# Interactive companion: opponent_season_rank vs total_points from feature matrix
st.markdown("**Interactive: Opponent Rank vs GW Points (xG era OOF)**")
st.caption(
    "Mean GW points by opponent season rank bucket across all xG era seasons. "
    "Loaded from the feature matrix parquet (all positions combined)."
)

fm_dfs = []
for pos in POSITIONS:
    path = FEATURES_DIR / f"feature_matrix_{pos}.parquet"
    if path.exists():
        fm = pd.read_parquet(path)[["opponent_season_rank", "total_points"]].copy()
        fm["position"] = pos
        fm_dfs.append(fm)

if fm_dfs:
    fm_all = pd.concat(fm_dfs, ignore_index=True).dropna(subset=["opponent_season_rank", "total_points"])
    fm_all["rank_bucket"] = pd.cut(
        fm_all["opponent_season_rank"],
        bins=[0, 3, 6, 10, 14, 17, 20],
        labels=["Top 3", "4–6", "7–10", "11–14", "15–17", "18–20"],
    )
    grp = (
        fm_all.groupby(["rank_bucket", "position"], observed=True)["total_points"]
        .mean()
        .round(3)
        .reset_index()
    )
    grp.columns = ["opponent_rank_bucket", "position", "mean_pts"]
    pos_order = [p for p in POSITIONS if p in grp["position"].unique()]
    grp["position"] = pd.Categorical(grp["position"], categories=pos_order, ordered=True)

    fig_fdr = px.bar(
        grp.sort_values(["position", "opponent_rank_bucket"]),
        x="opponent_rank_bucket",
        y="mean_pts",
        color="position",
        barmode="group",
        title="Mean GW Points by Opponent Rank Bucket",
        labels={"opponent_rank_bucket": "Opponent rank bucket", "mean_pts": "Mean GW pts"},
    )
    fig_fdr.update_layout(height=400)
    st.plotly_chart(fig_fdr, use_container_width=True)
else:
    st.info(
        "Feature matrix parquets not found. "
        "Clear `outputs/features/` and run `python run_gw.py` to regenerate."
    )

st.divider()

# ---------------------------------------------------------------------------
# Section D — Price vs Performance
# ---------------------------------------------------------------------------

st.subheader("Price vs Performance")

col1, col2 = st.columns(2)

price_path = EDA_DIR / "price_vs_season_points.png"
with col1:
    if price_path.exists():
        st.image(str(price_path), caption="Price vs season points", use_column_width=True)
    else:
        st.info("price_vs_season_points.png not found.")

band_path = EDA_DIR / "price_band_performance.png"
with col2:
    if band_path.exists():
        st.image(str(band_path), caption="Price band performance", use_column_width=True)
    else:
        st.info("price_band_performance.png not found.")

st.caption(
    "All costs stored as £0.1m units in the DB — divide by 10 for display. "
    "Premium players (>£11m) deliver more total points but lower points-per-million. "
    "Budget picks (<£5m) are high rotation risk — minutes are the binding constraint."
)

st.divider()

# ---------------------------------------------------------------------------
# Section E — Known Data Quirks
# ---------------------------------------------------------------------------

st.subheader("Known Data Quirks")

st.warning(
    "These quirks are handled in the ETL and ML pipeline. "
    "Be aware of them when writing custom SQL queries or interpreting raw data."
)

quirks = [
    {
        "Quirk": "2019-20 COVID gap",
        "Seasons affected": "Season 4",
        "Impact": (
            "GWs 30–38 never played. Season runs GW1–29 then GW39–47 (total_gws=47). "
            "Rolling windows must not cross the gap — handled in features.py."
        ),
    },
    {
        "Quirk": "2022-23 GW7 absent",
        "Seasons affected": "Season 7",
        "Impact": (
            "Postponed for Queen Elizabeth II national mourning; absorbed into other GWs. "
            "Season has 37 data GWs instead of 38."
        ),
    },
    {
        "Quirk": "Ferguson points discrepancy",
        "Seasons affected": "Season 9 (2024-25)",
        "Impact": (
            "1-pt and 17-min divergence between dim_player_season and fact_gw_player sums. "
            "Source data artefact — retroactive API correction not propagated to GW-level data."
        ),
    },
    {
        "Quirk": "Manager rows (2024-25)",
        "Seasons affected": "Season 9",
        "Impact": (
            "FPL manager game mode adds rows with value as low as 5 (£0.5m) and mng_* columns "
            "populated. Filtered by mng_win IS NULL in ML pipeline. position_label = 'AM'."
        ),
    },
    {
        "Quirk": "`starts` present in 2025-26",
        "Seasons affected": "Season 10",
        "Impact": (
            "Schema flags has_starts=0 for season 10 (defensive era), but data has starts "
            "populated across all 18,173 rows. season_starts_rate_to_date is computable."
        ),
    },
    {
        "Quirk": "`goals_conceded` is time-on-pitch scoped",
        "Seasons affected": "All seasons",
        "Impact": (
            "Reflects goals conceded while that specific player was on the pitch, not the team's "
            "match total. 75.9% of team-fixtures show inconsistent values across players. "
            "Never use for team-level defensive stats — derive from team_h_score/team_a_score."
        ),
    },
    {
        "Quirk": "`master_team_list.csv` missing 2024-25 and 2025-26",
        "Seasons affected": "Seasons 9–10",
        "Impact": (
            "Team mapping for recent seasons derived by cross-joining players_raw.team with "
            "merged_gw.team. Not a DB concern but relevant for raw CSV data access."
        ),
    },
]

quirks_df = pd.DataFrame(quirks)
st.dataframe(quirks_df, use_container_width=True, hide_index=True)
