"""Global Monitor page: top dislocations, region heatmaps, what changed today."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from vol_radar.config import load_config
from vol_radar.db.database import Database
from vol_radar.db.repos import DislocationRepo, FeatureRepo, InstrumentRepo
from dashboard.components.filters import history_toggle, region_filter
from dashboard.components.tables import format_leaderboard, style_dislocation_table

st.set_page_config(page_title="Global Monitor", layout="wide")
st.title("Global Monitor")

config = load_config()
db = Database(config.database.full_path)

# Sidebar filters
st.sidebar.header("Filters")
years = history_toggle(key="gm_history")

with db.get_session() as session:
    instruments = InstrumentRepo.get_all_instruments(session)
    inst_df = pd.DataFrame([
        {"ticker": i.ticker, "region": i.region, "instrument_id": i.instrument_id}
        for i in instruments if i.asset_class != "vol_index"
    ])

if inst_df.empty:
    st.warning("No instruments in database. Run the pipeline first.")
    st.stop()

selected_regions = region_filter(inst_df, key="gm_region")

# ── Dislocation Leaderboard ──
st.subheader("Dislocation Leaderboard (Latest)")

with db.get_session() as session:
    leaderboard = FeatureRepo.get_latest_features(session)

if not leaderboard.empty:
    leaderboard = leaderboard[leaderboard["region"].isin(selected_regions)]
    format_leaderboard(leaderboard)
else:
    st.info("No feature data available yet.")

# ── Region Heatmap ──
st.subheader("VRP Z-Score by Region")

if not leaderboard.empty and len(leaderboard) > 0:
    plot_df = leaderboard.copy()
    plot_df["composite_dislocation_score"] = plot_df["composite_dislocation_score"].clip(lower=0.1)

    # Pick the best available color metric (z_vrp_21d has data only for
    # instruments with a mapped vol index; fall back to composite score)
    vrp_coverage = plot_df["z_vrp_21d"].notna().mean()
    if vrp_coverage > 0.5:
        color_col, color_label = "z_vrp_21d", "VRP z-score"
    else:
        color_col, color_label = "composite_dislocation_score", "Dislocation Score"

    fig = px.treemap(
        plot_df,
        path=["region", "ticker"],
        values="composite_dislocation_score",
        color=color_col,
        color_continuous_scale="RdYlGn_r",
        color_continuous_midpoint=plot_df[color_col].median(),
        title=f"Dislocation Map (size=score, color={color_label})",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ── What Changed Today ──
st.subheader("Recent Dislocation Events")

with db.get_session() as session:
    events = DislocationRepo.get_latest_events(session, limit=30)

if not events.empty:
    style_dislocation_table(events)
else:
    st.info("No recent dislocation events.")

# ── Cross-Market Spreads ──
st.subheader("Cross-Market Z-Score Comparison")

if not leaderboard.empty and len(leaderboard) > 1:
    # Select z-score columns that have data for most instruments
    z_cols = ["z_vrp_21d", "z_curve_slope", "z_skew_proxy"]
    available_z = [c for c in z_cols if leaderboard[c].notna().sum() > 1]
    if not available_z:
        available_z = ["z_skew_proxy"]  # always has data

    spread_data = leaderboard[["ticker", "region"] + available_z].copy()
    spread_melted = spread_data.melt(
        id_vars=["ticker", "region"],
        var_name="metric",
        value_name="z_score",
    ).dropna(subset=["z_score"])

    label_map = {
        "z_vrp_21d": "VRP Z-Score",
        "z_curve_slope": "Curve Slope Z-Score",
        "z_skew_proxy": "Skew Proxy Z-Score",
    }
    spread_melted["metric"] = spread_melted["metric"].map(label_map).fillna(spread_melted["metric"])

    fig = px.bar(
        spread_melted,
        x="ticker", y="z_score", color="metric",
        barmode="group", facet_row="region",
        title="Z-Scores by Instrument and Region",
        color_discrete_map={
            "VRP Z-Score": "#2196F3",
            "Curve Slope Z-Score": "#FF9800",
            "Skew Proxy Z-Score": "#4CAF50",
        },
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
