"""Volatility feature calculations: RV, VRP, term structure, z-scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_variance(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Compute rolling realized variance.

    Args:
        returns: Log or simple returns series.
        window: Rolling window in days.
        annualize: If True, multiply by 252/window for annualization.

    Returns:
        Annualized realized variance series.
    """
    rv = returns.pow(2).rolling(window=window, min_periods=window).sum()
    if annualize:
        rv = rv * (252.0 / window)
    return rv


def realized_vol(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Compute rolling realized volatility (sqrt of realized variance).

    Args:
        returns: Returns series.
        window: Rolling window in days.
        annualize: If True, annualize the result.

    Returns:
        Annualized realized vol series (percentage scale, e.g. 0.15 = 15%).
    """
    rv = realized_variance(returns, window, annualize=annualize)
    return np.sqrt(rv)


def vrp(implied_vol: pd.Series, rv: pd.Series) -> pd.Series:
    """Compute variance risk premium (VRP).

    VRP = IV^2 - RV^2 (variance space)
    Positive VRP means implied vol exceeds realized — carry opportunity.

    Args:
        implied_vol: Implied vol series (decimal, e.g. 0.20 = 20%).
                    If vol index is on percentage scale (e.g., VIX = 20),
                    caller should divide by 100 first.
        rv: Realized vol series (same scale as implied_vol).

    Returns:
        VRP series in variance units.
    """
    return implied_vol.pow(2) - rv.pow(2)


def vrp_vol_level(implied_vol: pd.Series, rv: pd.Series) -> pd.Series:
    """Compute VRP in vol-level space (IV - RV).

    Args:
        implied_vol: Implied vol series.
        rv: Realized vol series.

    Returns:
        VRP in vol-level units.
    """
    return implied_vol - rv


def term_structure_slope(front: pd.Series, mid: pd.Series) -> pd.Series:
    """Compute term structure slope.

    Positive = contango (normal), Negative = backwardation (stress).

    Args:
        front: Front-month vol (e.g., VIX9D or VIX).
        mid: Mid-term vol (e.g., VIX3M or VXV).

    Returns:
        Slope series (mid - front).
    """
    return mid - front


def term_structure_curvature(
    front: pd.Series, mid: pd.Series, back: pd.Series
) -> pd.Series:
    """Compute term structure curvature (butterfly).

    Curvature = mid - 0.5 * (front + back)
    Positive = humped curve, Negative = inverted or flat.

    Args:
        front: Short-dated vol.
        mid: Medium-dated vol.
        back: Long-dated vol.

    Returns:
        Curvature series.
    """
    return mid - 0.5 * (front + back)


def roll_down_proxy(
    front: pd.Series, mid: pd.Series, days_to_expiry: int = 30
) -> pd.Series:
    """Compute daily roll-down proxy from contango.

    Estimates daily theta from the term structure slope.

    Args:
        front: Front-month vol.
        mid: Mid-term vol.
        days_to_expiry: Assumed days between front and mid tenors.

    Returns:
        Daily roll-down proxy.
    """
    return (mid - front) / days_to_expiry


def zscore(series: pd.Series, lookback: int = 252) -> pd.Series:
    """Compute rolling z-score.

    Args:
        series: Input series.
        lookback: Rolling window for mean/std calculation.

    Returns:
        Z-score series.
    """
    rolling_mean = series.rolling(window=lookback, min_periods=max(lookback // 2, 20)).mean()
    rolling_std = series.rolling(window=lookback, min_periods=max(lookback // 2, 20)).std()
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def drawdown(prices: pd.Series) -> pd.Series:
    """Compute running drawdown from peak.

    Args:
        prices: Price or equity curve series.

    Returns:
        Drawdown series (negative values, e.g. -0.10 = 10% drawdown).
    """
    peak = prices.expanding().max()
    dd = (prices - peak) / peak
    return dd
