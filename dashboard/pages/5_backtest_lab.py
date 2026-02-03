"""Backtest Lab page: strategy performance, attribution, trade log."""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from vol_radar.config import load_config
from vol_radar.db.database import Database
from vol_radar.db.repos import BacktestRepo, InstrumentRepo

st.set_page_config(page_title="Backtest Lab", layout="wide")
st.title("Backtest Lab")

config = load_config()
db = Database(config.database.full_path)

# Sidebar
st.sidebar.header("Filters")
from dashboard.components.filters import history_toggle

years = history_toggle(key="bt_history")

# Get available strategies
with db.get_session() as session:
    strategy_ids = BacktestRepo.get_strategy_ids(session)

if not strategy_ids:
    st.warning(
        "No backtest results available. Run the full pipeline first:\n\n"
        "```bash\npython -m vol_radar run --mode full\n```"
    )
    st.stop()

selected_strategy = st.sidebar.selectbox("Strategy", strategy_ids, key="bt_strategy")

# ── Load Performance Data ──
with db.get_session() as session:
    perf_df = BacktestRepo.get_perf(session, selected_strategy)
    trades_df = BacktestRepo.get_trades(session, selected_strategy)
    instruments = InstrumentRepo.get_all_instruments(session)
    instrument_regions = {i.ticker: i.region for i in instruments}

if perf_df.empty:
    st.info(f"No performance data for strategy '{selected_strategy}'.")
    st.stop()

# ── Performance Metrics ──
st.subheader("Performance Metrics")

from vol_radar.backtest.performance import PerformanceAttributor

pa = PerformanceAttributor()
equity = pa.compute_equity_curve(perf_df["pnl"], config.backtest.initial_capital)
dd_series = pa.compute_drawdown_series(equity)
metrics = pa.compute_metrics(equity)

# Key metrics in columns
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Sharpe", f"{metrics['sharpe']:.2f}")
with col2:
    st.metric("Sortino", f"{metrics['sortino']:.2f}")
with col3:
    st.metric("Max DD", f"{metrics['max_drawdown']:.1%}")
with col4:
    st.metric("Hit Rate", f"{metrics['hit_rate']:.1%}")
with col5:
    st.metric("Total Return", f"{metrics['total_return']:.1%}")

# Full metrics table
from dashboard.components.tables import style_metrics_table

with st.expander("Full Metrics"):
    style_metrics_table(metrics)

# ── Equity Curve + Drawdown ──
st.subheader("Equity Curve & Drawdown")

from dashboard.components.charts import plot_equity_curve

fig = plot_equity_curve(
    equity, dd_series, dates=perf_df["date"],
    title=f"Strategy: {selected_strategy}",
)
st.plotly_chart(fig, use_container_width=True)

# ── Rolling Sharpe ──
st.subheader("Rolling Performance")

rolling_window = st.slider("Rolling Window (days)", 21, 252, 63, key="bt_rolling")
rolling = pa.rolling_metrics(equity, window=rolling_window)

from dashboard.components.charts import plot_time_series

col1, col2 = st.columns(2)
with col1:
    rolling_df = pd.DataFrame({
        "date": perf_df["date"],
        "rolling_sharpe": rolling["rolling_sharpe"],
    })
    fig = plot_time_series(
        rolling_df, "date", ["rolling_sharpe"],
        title=f"Rolling Sharpe ({rolling_window}D)",
        y_title="Sharpe Ratio",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    rolling_df["rolling_vol"] = rolling["rolling_vol"]
    fig = plot_time_series(
        rolling_df, "date", ["rolling_vol"],
        title=f"Rolling Volatility ({rolling_window}D)",
        y_title="Annualized Vol",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PnL Attribution ──
st.subheader("PnL Attribution")

col1, col2 = st.columns(2)

# By Region
with col1:
    if not trades_df.empty and "ticker" in trades_df.columns:
        trades_with_region = trades_df.copy()
        trades_with_region["region"] = trades_with_region["ticker"].map(instrument_regions).fillna("Unknown")

        region_pnl = trades_with_region.groupby("region")["pnl"].sum().reset_index()

        if not region_pnl.empty:
            from dashboard.components.charts import plot_bar_attribution
            fig = plot_bar_attribution(region_pnl, "region", "pnl", title="PnL by Region")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade PnL data for region attribution.")
    else:
        st.info("No trades data available.")

# By Instrument
with col2:
    if not trades_df.empty and "ticker" in trades_df.columns:
        ticker_pnl = trades_df.groupby("ticker")["pnl"].sum().reset_index()
        ticker_pnl = ticker_pnl.sort_values("pnl", ascending=False)

        if not ticker_pnl.empty:
            from dashboard.components.charts import plot_bar_attribution
            fig = plot_bar_attribution(ticker_pnl, "ticker", "pnl", title="PnL by Instrument")
            st.plotly_chart(fig, use_container_width=True)

# ── Trade Log ──
st.subheader("Trade Log")

if not trades_df.empty:
    # Format for display
    display_trades = trades_df.copy()
    for col in ["signal", "position", "fill_price", "pnl", "fees", "slippage"]:
        if col in display_trades.columns:
            display_trades[col] = display_trades[col].round(4)

    st.dataframe(
        display_trades.sort_values("date", ascending=False).head(200),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No trade data available.")

# ── Upload CSV for Vol Index Data ──
st.subheader("Upload Vol Index Data")
st.markdown(
    "Upload a CSV file with columns: `date`, `ticker`, `vol_level`, `source`, `quality_flag`"
)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="bt_upload")
if uploaded_file is not None:
    try:
        from vol_radar.ingest.upload import CsvUploadHandler

        handler = CsvUploadHandler()
        df = handler.parse_from_streamlit_upload(uploaded_file)
        st.success(f"Parsed {len(df)} rows from upload.")
        st.dataframe(df.head(20), use_container_width=True)

        if st.button("Store in Database", key="bt_store"):
            with db.get_session() as session:
                count = handler.store_uploaded_data(df, session)
            st.success(f"Stored {count} rows. Re-run pipeline to recompute features.")
    except Exception as e:
        st.error(f"Upload error: {e}")
