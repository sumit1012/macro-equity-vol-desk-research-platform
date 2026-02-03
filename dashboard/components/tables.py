"""Reusable table display components."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def style_dislocation_table(df: pd.DataFrame) -> None:
    """Display dislocation events with severity color coding."""
    if df.empty:
        st.info("No dislocation events found.")
        return

    # Color map for severity
    def severity_color(val):
        colors = {
            "extreme": "background-color: #F44336; color: white",
            "high": "background-color: #FF9800; color: white",
            "medium": "background-color: #FFC107; color: black",
            "low": "background-color: #4CAF50; color: white",
        }
        return colors.get(str(val).lower(), "")

    display_df = df.copy()
    if "severity" in display_df.columns:
        display_df["severity"] = display_df["severity"].round(1)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def style_metrics_table(metrics: dict) -> None:
    """Display performance metrics in a formatted table."""
    if not metrics:
        st.info("No metrics available.")
        return

    formatted = {}
    for key, value in metrics.items():
        if key in ("sharpe", "sortino", "calmar"):
            formatted[key.title()] = f"{value:.2f}"
        elif key in ("max_drawdown", "hit_rate", "annualized_return", "total_return", "daily_vol"):
            formatted[key.replace("_", " ").title()] = f"{value:.2%}"
        elif key in ("cvar_95",):
            formatted["CVaR (95%)"] = f"{value:.4f}"
        elif key == "n_days":
            formatted["Trading Days"] = f"{value:,}"
        else:
            formatted[key.replace("_", " ").title()] = f"{value:.4f}"

    df = pd.DataFrame(
        list(formatted.items()), columns=["Metric", "Value"]
    )
    st.table(df)


def format_leaderboard(df: pd.DataFrame) -> None:
    """Display global dislocation leaderboard with conditional formatting."""
    if df.empty:
        st.info("No leaderboard data available.")
        return

    display_df = df.copy()

    # Format numeric columns
    for col in ["vrp_21d", "curve_slope", "skew_proxy"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    for col in ["z_vrp_21d", "z_curve_slope", "z_skew_proxy", "composite_dislocation_score", "regime_score"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "composite_dislocation_score": st.column_config.ProgressColumn(
                "Dislocation Score",
                help="Composite dislocation score (0-100)",
                min_value=0,
                max_value=100,
            ),
            "regime_score": st.column_config.ProgressColumn(
                "Regime Score",
                help="Regime score (0=calm, 100=panic)",
                min_value=0,
                max_value=100,
            ),
        },
    )
