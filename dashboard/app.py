"""Streamlit dashboard entry point for Vol Radar."""

import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Vol Dislocation Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Global Equity Volatility Dislocation Radar")
st.markdown("---")

# Check database status
from vol_radar.config import load_config
from vol_radar.db.database import Database
from vol_radar.db.repos import ManifestRepo, InstrumentRepo

try:
    config = load_config()
    db_path = config.database.full_path

    if not db_path.exists():
        st.warning(
            "Database not initialized. Run the pipeline first:\n\n"
            "```bash\npython -m vol_radar run --mode full\n```"
        )
        st.stop()

    db = Database(db_path)

    with db.get_session() as session:
        latest_run = ManifestRepo.get_latest_run(session)
        run_timestamp = latest_run.timestamp.strftime("%Y-%m-%d %H:%M") if latest_run else None
        run_status = latest_run.status if latest_run else None

        instruments = InstrumentRepo.get_all_instruments(session)
        equity_count = sum(1 for i in instruments if i.asset_class != "vol_index")
        vol_count = sum(1 for i in instruments if i.asset_class == "vol_index")

    # Dashboard overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Equity ETFs", equity_count)
    with col2:
        st.metric("Vol Indices", vol_count)
    with col3:
        st.metric("Last Run", run_timestamp or "Never")
    with col4:
        if run_status:
            status_color = "🟢" if run_status == "completed" else "🔴"
            st.metric("Status", f"{status_color} {run_status}")
        else:
            st.metric("Status", "No runs")

    st.markdown("---")
    st.markdown(
        """
        ### Navigation
        Use the sidebar to navigate between pages:
        - **Global Monitor** - Top dislocations, region heatmaps
        - **Vol Term Structure** - Curves by region, slope/curvature
        - **VRP & Skew Lab** - VRP bands, skew proxies, scatter analysis
        - **Dislocation Alerts** - Rules triggered, severity drill-down
        - **Backtest Lab** - Strategy performance, attribution, trade log
        """
    )

except FileNotFoundError:
    st.error("Config file not found. Ensure config/default.yaml exists.")
except Exception as e:
    st.error(f"Error loading dashboard: {e}")
