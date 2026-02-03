"""VRP & Skew Lab page: VRP bands, skew proxies, scatter analysis."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from vol_radar.config import load_config
from vol_radar.db.database import Database
from vol_radar.db.repos import FeatureRepo, InstrumentRepo

st.set_page_config(page_title="VRP & Skew Lab", layout="wide")
st.title("VRP & Skew Lab")

config = load_config()
db = Database(config.database.full_path)

# Sidebar
st.sidebar.header("Filters")
from dashboard.components.filters import history_toggle, horizon_filter

years = history_toggle(key="vrp_history")
horizon = horizon_filter(key="vrp_horizon")
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

selected_tickers = st.sidebar.multiselect(
    "Instruments",
    inst_df["ticker"].tolist(),
    default=["SPY"] if "SPY" in inst_df["ticker"].values else inst_df["ticker"].tolist()[:3],
    key="vrp_instruments",
)

# Map horizon to column
horizon_map = {"5D": "5d", "21D": "21d", "63D": "63d"}
h = horizon_map.get(horizon, "21d")

for ticker in selected_tickers:
    st.subheader(f"{ticker}")

    with db.get_session() as session:
        inst_id = InstrumentRepo.get_instrument_id(session, ticker)
        if inst_id is None:
            st.warning(f"Instrument {ticker} not found.")
            continue

        features = FeatureRepo.get_features(
            session, instrument_ids=[inst_id],
            start_date=start_date, end_date=end_date,
        )

    if features.empty:
        st.info(f"No feature data for {ticker}.")
        continue

    col1, col2 = st.columns(2)

    # ── VRP with Bands ──
    with col1:
        vrp_col = f"vrp_{h}"
        if vrp_col in features.columns and features[vrp_col].notna().any():
            from dashboard.components.charts import plot_vrp_bands
            fig = plot_vrp_bands(
                features, date_col="date", vrp_col=vrp_col,
                title=f"VRP ({horizon}) - {ticker}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No VRP data for {ticker} (may need vol index upload).")

    # ── Skew Proxy ──
    with col2:
        if features["skew_proxy"].notna().any():
            from dashboard.components.charts import plot_time_series
            fig = plot_time_series(
                features, "date", ["skew_proxy", "vol_of_vol"],
                title=f"Skew Proxy & Vol-of-Vol - {ticker}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No skew data for {ticker}.")

    # ── Realized Vol Comparison ──
    rv_cols = [f"rv_{w}d" for w in [5, 21, 63] if f"rv_{w}d" in features.columns]
    if rv_cols:
        from dashboard.components.charts import plot_time_series
        fig = plot_time_series(
            features, "date", rv_cols,
            title=f"Realized Volatility - {ticker}",
            y_title="Annualized Vol",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── VRP vs Forward Returns Scatter ──
    vrp_col = f"vrp_{h}"
    if vrp_col in features.columns and features[vrp_col].notna().any():
        # Compute forward returns
        features_plot = features.copy()
        features_plot["fwd_return_21d"] = features_plot[f"rv_{h}"].shift(-21)

        valid = features_plot.dropna(subset=[vrp_col, "fwd_return_21d"])
        if len(valid) > 20:
            from dashboard.components.charts import plot_scatter
            fig = plot_scatter(
                valid, vrp_col, "fwd_return_21d",
                title=f"VRP vs Forward RV - {ticker}",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
