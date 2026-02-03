"""Dislocation Alerts page: triggered rules, severity drill-down."""

import json
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
from vol_radar.db.repos import DislocationRepo, InstrumentRepo

st.set_page_config(page_title="Dislocation Alerts", layout="wide")
st.title("Dislocation Alerts")

config = load_config()
db = Database(config.database.full_path)

# Sidebar filters
st.sidebar.header("Filters")
from dashboard.components.filters import history_toggle, severity_filter

years = history_toggle(key="da_history")
min_severity = severity_filter(key="da_severity")
end_date = date.today()
start_date = end_date - timedelta(days=365 * years)

with db.get_session() as session:
    instruments = InstrumentRepo.get_all_instruments(session)
    tickers = [i.ticker for i in instruments if i.asset_class != "vol_index"]

selected_ticker = st.sidebar.selectbox(
    "Instrument (or All)", ["All"] + tickers, key="da_ticker"
)

event_types = st.sidebar.multiselect(
    "Event Types",
    ["vrp_extreme", "curve_inversion", "curve_dislocation", "skew_extreme"],
    default=["vrp_extreme", "curve_inversion", "curve_dislocation", "skew_extreme"],
    key="da_event_types",
)

# ── Fetch Events ──
with db.get_session() as session:
    ticker_filter = selected_ticker if selected_ticker != "All" else None
    events = DislocationRepo.get_events(
        session, ticker=ticker_filter,
        start_date=start_date, end_date=end_date,
        limit=500,
    )

if events.empty:
    st.info("No dislocation events found for the selected filters.")
    st.stop()

# Filter by severity and event type
if min_severity > 0:
    events = events[events["severity"] >= min_severity]
if event_types:
    events = events[events["event_type"].isin(event_types)]

if events.empty:
    st.info("No events match the current filters.")
    st.stop()

# ── Active Alerts Table ──
st.subheader(f"Active Alerts ({len(events)} events)")
from dashboard.components.tables import style_dislocation_table

display_events = events[["date", "ticker", "region", "event_type", "severity"]].copy()
display_events["severity"] = display_events["severity"].round(1)
display_events = display_events.sort_values("date", ascending=False)
style_dislocation_table(display_events)

# ── Alert History Chart ──
st.subheader("Alert History")

events_by_date = events.copy()
events_by_date["date"] = pd.to_datetime(events_by_date["date"])
events_by_date["month"] = events_by_date["date"].dt.to_period("M").astype(str)

monthly_counts = events_by_date.groupby(["month", "event_type"]).size().reset_index(name="count")

fig = px.bar(
    monthly_counts,
    x="month", y="count", color="event_type",
    title="Monthly Dislocation Event Count",
    barmode="stack",
    color_discrete_map={
        "vrp_extreme": "#2196F3",
        "curve_inversion": "#FF9800",
        "curve_dislocation": "#FF9800",
        "skew_extreme": "#4CAF50",
    },
)
fig.update_layout(height=400, template="plotly_white", xaxis_title="Month", yaxis_title="Event Count")
st.plotly_chart(fig, use_container_width=True)

# ── Severity Distribution ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Severity Distribution")
    fig = px.histogram(
        events, x="severity", nbins=20,
        title="Distribution of Severity Scores",
        color_discrete_sequence=["#2196F3"],
    )
    fig.update_layout(height=350, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Events by Type")
    type_counts = events["event_type"].value_counts().reset_index()
    type_counts.columns = ["event_type", "count"]
    fig = px.pie(type_counts, values="count", names="event_type", title="Event Type Breakdown")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── Event Detail Drill-Down ──
st.subheader("Event Details")

selected_idx = st.selectbox(
    "Select an event to drill down",
    range(len(events)),
    format_func=lambda i: f"{events.iloc[i]['date']} | {events.iloc[i]['ticker']} | {events.iloc[i]['event_type']} (severity: {events.iloc[i]['severity']:.1f})",
    key="da_drill_down",
)

if selected_idx is not None:
    selected_event = events.iloc[selected_idx]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Date:** {selected_event['date']}")
        st.markdown(f"**Ticker:** {selected_event['ticker']}")
        st.markdown(f"**Region:** {selected_event['region']}")

    with col2:
        st.markdown(f"**Event Type:** {selected_event['event_type']}")
        st.markdown(f"**Severity:** {selected_event['severity']:.1f}")

    if selected_event.get("payload_json"):
        try:
            payload = json.loads(selected_event["payload_json"])
            st.json(payload)
        except (json.JSONDecodeError, TypeError):
            st.text(str(selected_event.get("payload_json", "")))
