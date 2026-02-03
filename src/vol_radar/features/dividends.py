"""Dividend shock detection via price gap analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_ex_div_dates(
    close: pd.Series,
    adj_close: pd.Series,
    threshold: float = 0.001,
) -> pd.Series:
    """Detect ex-dividend dates using close vs adjusted close divergence.

    On ex-div dates, the ratio adj_close/close changes because the adjustment
    factor shifts. We detect this as a proxy for ex-div dates.

    Args:
        close: Raw close prices.
        adj_close: Adjusted close prices.
        threshold: Minimum ratio change to flag as ex-div.

    Returns:
        Boolean series (True = likely ex-div date).
    """
    # Compute adjustment ratio
    ratio = adj_close / close.replace(0, np.nan)
    ratio_change = ratio.diff().abs()

    # Flag days where adjustment ratio changed significantly
    is_ex_div = ratio_change > threshold

    return is_ex_div.fillna(False)


def dividend_shock_window(
    ex_div_dates: pd.Series,
    window: int = 5,
) -> pd.Series:
    """Flag a window of days around each ex-dividend date.

    Args:
        ex_div_dates: Boolean series of ex-div dates.
        window: Number of days before and after to flag.

    Returns:
        Boolean series (True = within window of ex-div).
    """
    # Convert to float for rolling
    ex_div_float = ex_div_dates.astype(float)

    # Flag forward window (look back: was there an ex-div in the last `window` days?)
    forward = ex_div_float.rolling(window=window, min_periods=1).max()

    # Flag backward window (look ahead: is there an ex-div in the next `window` days?)
    backward = ex_div_float[::-1].rolling(window=window, min_periods=1).max()[::-1]

    in_window = (forward > 0) | (backward > 0)
    return in_window.astype(bool)


def dividend_gap_magnitude(
    open_prices: pd.Series,
    prev_close: pd.Series,
) -> pd.Series:
    """Compute overnight gap magnitude (open vs previous close).

    Useful for measuring the price impact of dividends.

    Args:
        open_prices: Open price series.
        prev_close: Previous day's close price (close.shift(1)).

    Returns:
        Gap magnitude as fraction (e.g., -0.02 = 2% gap down).
    """
    prev_close_safe = prev_close.replace(0, np.nan)
    return (open_prices - prev_close) / prev_close_safe


def div_shock_flag(
    gap_magnitude: pd.Series,
    ex_div_dates: pd.Series,
    threshold: float = 0.02,
) -> pd.Series:
    """Flag significant dividend-related price gaps.

    Args:
        gap_magnitude: Overnight gap series.
        ex_div_dates: Boolean series of ex-div dates.
        threshold: Minimum absolute gap to flag.

    Returns:
        Boolean series (True = significant div shock).
    """
    is_shock = ex_div_dates & (gap_magnitude.abs() > threshold)
    return is_shock.fillna(False)
