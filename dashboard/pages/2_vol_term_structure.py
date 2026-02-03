"""Vol Term Structure page: curves by region, slope/curvature time series."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from vol_radar.config import load_config
from vol_radar.db.database import Database
from vol_radar.db.repos import FeatureRepo, InstrumentRepo, VolIndexRepo
from dashboard.components.filters import history_toggle, region_filter
from dashboard.components.charts import plot_time_series

st.set_page_config(page_title="Vol Term Structure", layout="wide")
st.title("Vol Term Structure")

config = load_config()
db = Database(config.database.full_path)

# Sidebar
st.sidebar.header("Filters")
years = history_toggle(key="vts_history")
end_date = date.today()
start_date = end_date - timedelta(days=365 * years)

with db.get_session() as session:
    instruments = InstrumentRepo.get_all_instruments(session)
    inst_df = pd.DataFrame([
        {"ticker": i.ticker, "region": i.region, "instrument_id": i.instrument_id}
        for i in instruments if i.asset_class != "vol_index"
    ])

if inst_df.empty:
    st.warning("No instruments found.")
    st.stop()

selected_regions = region_filter(inst_df, key="vts_region")

# ── Current Term Structure Snapshot ──
st.subheader("Current Vol Index Levels")

with db.get_session() as session:
    vol_indices_data = {}
    for vi in config.vol_indices:
        inst_id = InstrumentRepo.get_instrument_id(session, vi.ticker)
        if inst_id:
            vol_df = VolIndexRepo.get_vol_data(session, inst_id, start_date, end_date)
            if not vol_df.empty:
                vol_indices_data[vi.ticker] = vol_df

if vol_indices_data:
    # Show latest values
    latest_vals = {}
    for ticker, df in vol_indices_data.items():
        if not df.empty:
            latest_vals[ticker] = df["vol_level"].iloc[-1]

    if latest_vals:
        cols = st.columns(len(latest_vals))
        for col, (ticker, val) in zip(cols, latest_vals.items()):
            with col:
                st.metric(ticker, f"{val:.1f}")

    # Term structure chart for US VIX family
    us_vix_tickers = ["^VIX9D", "^VIX", "^VIX3M"]
    us_vix_data = {t: vol_indices_data[t] for t in us_vix_tickers if t in vol_indices_data}

    if len(us_vix_data) >= 2:
        st.subheader("US VIX Term Structure (Latest)")
        latest_points = {}
        for ticker in us_vix_tickers:
            if ticker in us_vix_data:
                df = us_vix_data[ticker]
                latest_points[ticker] = df["vol_level"].iloc[-1]

        if latest_points:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(latest_points.keys()),
                y=list(latest_points.values()),
                mode="lines+markers",
                marker=dict(size=12),
                line=dict(width=3),
            ))
            fig.update_layout(
                yaxis_title="Vol Level",
                height=300,
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No vol index data available. Upload or ingest vol index data first.")

# ── Slope & Curvature Time Series ──
st.subheader("Term Structure Slope & Curvature")

with db.get_session() as session:
    filtered_ids = inst_df[inst_df["region"].isin(selected_regions)]["instrument_id"].tolist()
    features = FeatureRepo.get_features(session, instrument_ids=filtered_ids, start_date=start_date, end_date=end_date)

if not features.empty:
    # Select representative instruments
    selected_ticker = st.selectbox(
        "Instrument",
        inst_df[inst_df["region"].isin(selected_regions)]["ticker"].tolist(),
        key="vts_ticker",
    )

    with db.get_session() as session:
        sel_id = InstrumentRepo.get_instrument_id(session, selected_ticker)

    if sel_id:
        inst_features = features[features["instrument_id"] == sel_id]

        if not inst_features.empty:
            col1, col2 = st.columns(2)
            with col1:
                if inst_features["curve_slope"].notna().any():
                    fig = plot_time_series(
                        inst_features, "date", ["curve_slope"],
                        title=f"Curve Slope - {selected_ticker}",
                        y_title="Slope",
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No curve slope data for this instrument.")

            with col2:
                if inst_features["curve_curvature"].notna().any():
                    fig = plot_time_series(
                        inst_features, "date", ["curve_curvature"],
                        title=f"Curve Curvature - {selected_ticker}",
                        y_title="Curvature",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No curvature data for this instrument.")

            # Contango/Backwardation indicator
            if inst_features["curve_slope"].notna().any():
                latest_slope = inst_features["curve_slope"].dropna().iloc[-1]
                if latest_slope > 0:
                    st.success(f"Term structure in **CONTANGO** (slope: {latest_slope:.2f})")
                else:
                    st.error(f"Term structure in **BACKWARDATION** (slope: {latest_slope:.2f})")
else:
    st.info("No feature data available.")
