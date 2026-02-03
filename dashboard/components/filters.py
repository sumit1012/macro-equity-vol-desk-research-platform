"""Reusable Streamlit filter widgets."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st
import pandas as pd


def region_filter(instruments_df: pd.DataFrame, key: str = "region") -> list[str]:
    """Multi-select for region filtering."""
    regions = sorted(instruments_df["region"].unique())
    return st.sidebar.multiselect(
        "Region", regions, default=regions, key=f"filter_{key}"
    )


def instrument_filter(
    instruments_df: pd.DataFrame,
    selected_regions: list[str],
    key: str = "instrument",
) -> list[str]:
    """Multi-select for instrument filtering, pre-filtered by region."""
    filtered = instruments_df[instruments_df["region"].isin(selected_regions)]
    tickers = sorted(filtered["ticker"].unique())
    return st.sidebar.multiselect(
        "Instruments", tickers, default=tickers, key=f"filter_{key}"
    )


def date_range_filter(
    default_years: int = 5, key: str = "date_range"
) -> tuple[date, date]:
    """Date range selector in sidebar."""
    end = date.today()
    start = end - timedelta(days=365 * default_years)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start", value=start, key=f"filter_{key}_start")
    with col2:
        end_date = st.date_input("End", value=end, key=f"filter_{key}_end")

    return start_date, end_date


def history_toggle(key: str = "history") -> int:
    """History length toggle: 1Y, 2Y, 5Y, Max."""
    options = {"1Y": 1, "2Y": 2, "5Y": 5, "Max": 20}
    selected = st.sidebar.radio(
        "History", list(options.keys()), index=2, key=f"filter_{key}",
        horizontal=True,
    )
    return options[selected]


def horizon_filter(key: str = "horizon") -> str:
    """Selectbox for vol horizon."""
    return st.sidebar.selectbox(
        "Horizon", ["5D", "21D", "63D"], index=1, key=f"filter_{key}"
    )


def regime_filter(key: str = "regime") -> list[str]:
    """Multi-select for regime filtering."""
    all_regimes = ["risk_off", "carry_friendly", "neutral", "transition"]
    return st.sidebar.multiselect(
        "Regime", all_regimes, default=all_regimes, key=f"filter_{key}"
    )


def severity_filter(key: str = "severity") -> float:
    """Severity threshold slider."""
    return st.sidebar.slider(
        "Min Severity Score", min_value=0.0, max_value=100.0,
        value=40.0, step=5.0, key=f"filter_{key}"
    )
