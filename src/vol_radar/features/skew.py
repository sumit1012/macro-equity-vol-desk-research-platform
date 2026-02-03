"""Skew proxy calculations: downside tail intensity, vol-of-vol."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vol_radar.features.volatility import zscore


def downside_tail_intensity(
    returns: pd.Series,
    window: int = 63,
    threshold_std: float = -2.0,
) -> pd.Series:
    """Compute rolling downside tail intensity.

    Measures the fraction of days within the rolling window where returns
    fall below `threshold_std` standard deviations.

    Args:
        returns: Daily returns series.
        window: Rolling window in days.
        threshold_std: Number of standard deviations below zero to count as tail event.

    Returns:
        Series of tail intensities (0 to 1).
    """
    rolling_std = returns.rolling(window=window, min_periods=window // 2).std()
    threshold = threshold_std * rolling_std

    # Boolean: is this day a tail event?
    is_tail = (returns < threshold).astype(float)

    # Rolling count of tail events, normalized by window
    intensity = is_tail.rolling(window=window, min_periods=window // 2).mean()
    return intensity


def vol_of_vol(rv_series: pd.Series, window: int = 21) -> pd.Series:
    """Compute volatility of volatility (vol-of-vol).

    Rolling standard deviation of realized vol, capturing how unstable
    the vol environment is.

    Args:
        rv_series: Realized volatility series.
        window: Rolling window for computing std of vol.

    Returns:
        Vol-of-vol series.
    """
    return rv_series.rolling(window=window, min_periods=window // 2).std()


def skew_proxy(
    downside: pd.Series,
    vov: pd.Series,
    weights: tuple[float, float] = (0.5, 0.5),
    zscore_lookback: int = 252,
) -> pd.Series:
    """Compute composite skew proxy.

    Combines downside tail intensity and vol-of-vol into a single
    skew indicator, z-scored for comparability.

    Args:
        downside: Downside tail intensity series.
        vov: Vol-of-vol series.
        weights: Weights for (downside, vov) components.
        zscore_lookback: Lookback for z-scoring components.

    Returns:
        Composite skew proxy (z-scored).
    """
    # Z-score each component
    z_downside = zscore(downside, lookback=zscore_lookback)
    z_vov = zscore(vov, lookback=zscore_lookback)

    # Weighted combination
    composite = weights[0] * z_downside.fillna(0) + weights[1] * z_vov.fillna(0)

    return composite
