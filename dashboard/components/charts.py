"""Reusable Plotly chart components for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_time_series(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    title: str = "",
    y_title: str = "",
    height: int = 400,
) -> go.Figure:
    """Multi-line time series chart."""
    fig = go.Figure()
    for col in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x], y=df[col], name=col, mode="lines",
            ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title=y_title,
        height=height, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    title: str = "",
    height: int = 400,
    colorscale: str = "RdYlGn_r",
) -> go.Figure:
    """Heatmap chart (e.g., cross-market spreads)."""
    pivot = df.pivot_table(values=z, index=y, columns=x, aggfunc="last")
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscale,
        texttemplate="%{z:.2f}",
    ))
    fig.update_layout(
        title=title, height=height, template="plotly_white",
    )
    return fig


def plot_equity_curve(
    equity: pd.Series,
    drawdown: pd.Series,
    dates: pd.Series | None = None,
    title: str = "Equity Curve & Drawdown",
    height: int = 500,
) -> go.Figure:
    """Dual-axis equity curve + drawdown chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown"),
    )

    x_axis = dates if dates is not None else equity.index

    fig.add_trace(
        go.Scatter(x=x_axis, y=equity, name="Equity", line=dict(color="#2196F3")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis, y=drawdown, name="Drawdown",
            fill="tozeroy", line=dict(color="#F44336"),
        ),
        row=2, col=1,
    )

    fig.update_layout(height=height, template="plotly_white", title=title)
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str = "",
    height: int = 400,
) -> go.Figure:
    """Scatter plot with optional color dimension."""
    fig = go.Figure()

    if color and color in df.columns:
        for cat in df[color].unique():
            subset = df[df[color] == cat]
            fig.add_trace(go.Scatter(
                x=subset[x], y=subset[y], name=str(cat),
                mode="markers", marker=dict(size=6, opacity=0.6),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y], mode="markers",
            marker=dict(size=6, opacity=0.6, color="#2196F3"),
        ))

    fig.update_layout(
        title=title, xaxis_title=x, yaxis_title=y,
        height=height, template="plotly_white",
    )
    return fig


def plot_bar_attribution(
    df: pd.DataFrame,
    category: str,
    value: str,
    title: str = "",
    height: int = 350,
) -> go.Figure:
    """Bar chart for PnL attribution."""
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in df[value]]
    fig = go.Figure(data=go.Bar(
        x=df[category], y=df[value],
        marker_color=colors,
    ))
    fig.update_layout(
        title=title, yaxis_title=value,
        height=height, template="plotly_white",
    )
    return fig


def plot_term_structure_curves(
    data: dict[str, pd.Series],
    title: str = "Vol Term Structure",
    height: int = 400,
) -> go.Figure:
    """Plot term structure curves for multiple regions/dates."""
    fig = go.Figure()
    for label, series in data.items():
        fig.add_trace(go.Scatter(
            x=series.index.tolist(), y=series.values,
            name=label, mode="lines+markers",
        ))
    fig.update_layout(
        title=title, xaxis_title="Tenor", yaxis_title="Vol Level",
        height=height, template="plotly_white",
    )
    return fig


def plot_vrp_bands(
    df: pd.DataFrame,
    date_col: str = "date",
    vrp_col: str = "vrp_21d",
    title: str = "VRP with Bands",
    height: int = 400,
    lookback: int = 252,
) -> go.Figure:
    """VRP time series with +/- 1 and 2 std bands."""
    fig = go.Figure()

    rolling_mean = df[vrp_col].rolling(lookback).mean()
    rolling_std = df[vrp_col].rolling(lookback).std()

    # 2-std bands
    fig.add_trace(go.Scatter(
        x=df[date_col], y=rolling_mean + 2 * rolling_std,
        name="+2σ", line=dict(dash="dot", color="rgba(255,0,0,0.3)"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col], y=rolling_mean - 2 * rolling_std,
        name="-2σ", fill="tonexty",
        line=dict(dash="dot", color="rgba(255,0,0,0.3)"),
        fillcolor="rgba(255,0,0,0.05)",
        showlegend=False,
    ))

    # 1-std bands
    fig.add_trace(go.Scatter(
        x=df[date_col], y=rolling_mean + rolling_std,
        name="+1σ", line=dict(dash="dash", color="rgba(0,0,255,0.3)"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col], y=rolling_mean - rolling_std,
        name="-1σ", fill="tonexty",
        line=dict(dash="dash", color="rgba(0,0,255,0.3)"),
        fillcolor="rgba(0,0,255,0.05)",
        showlegend=False,
    ))

    # VRP line
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[vrp_col],
        name="VRP", line=dict(color="#2196F3", width=1.5),
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title, height=height, template="plotly_white",
        yaxis_title="VRP (Variance)",
    )
    return fig
