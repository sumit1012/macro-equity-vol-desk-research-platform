"""ETF flow proxy calculations: dollar volume shocks, impact proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vol_radar.features.volatility import zscore


def dollar_volume(price: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute dollar volume (price * volume).

    Args:
        price: Close price series.
        volume: Volume series.

    Returns:
        Dollar volume series.
    """
    return price * volume


def dollar_volume_shock(
    dv: pd.Series, lookback: int = 63, method: str = "zscore"
) -> pd.Series:
    """Compute dollar volume shock indicator.

    Detects unusual trading activity relative to trailing median.

    Args:
        dv: Dollar volume series.
        lookback: Trailing window for computing median.
        method: 'zscore' or 'ratio' (vs median).

    Returns:
        Shock indicator (z-score or ratio).
    """
    if method == "zscore":
        return zscore(dv, lookback=lookback)
    else:
        rolling_median = dv.rolling(window=lookback, min_periods=lookback // 2).median()
        rolling_median = rolling_median.replace(0, np.nan)
        return dv / rolling_median


def impact_proxy(returns: pd.Series, dv: pd.Series) -> pd.Series:
    """Compute price impact proxy.

    Measures how much price moves per unit of dollar volume.
    High impact = low liquidity or unusual flow.

    Args:
        returns: Absolute returns series.
        dv: Dollar volume series.

    Returns:
        Impact proxy (abs(return) / log(dollar_volume)).
    """
    log_dv = np.log(dv.clip(lower=1))
    log_dv = log_dv.replace(0, np.nan)
    return returns.abs() / log_dv


def flow_regime_flag(
    dv_shock: pd.Series, threshold: float = 2.0
) -> pd.Series:
    """Flag days with extreme dollar volume shocks.

    Args:
        dv_shock: Dollar volume shock z-score.
        threshold: Z-score threshold for flagging.

    Returns:
        Boolean series (True = shock day).
    """
    return dv_shock.abs() > threshold
