"""Shared test fixtures for Vol Radar tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from vol_radar.config import load_config
from vol_radar.db.database import Database


@pytest.fixture
def db():
    """Create an in-memory database with all tables."""
    database = Database(":memory:")
    database.create_tables()
    return database


@pytest.fixture
def config():
    """Load the default configuration."""
    return load_config()


@pytest.fixture
def sample_dates() -> list[date]:
    """Generate 500 trading dates."""
    start = date(2020, 1, 2)
    dates = []
    current = start
    while len(dates) < 500:
        if current.weekday() < 5:  # Skip weekends
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture
def sample_prices(sample_dates) -> pd.DataFrame:
    """Generate synthetic OHLCV price data for SPY."""
    np.random.seed(42)
    n = len(sample_dates)

    # Random walk for close prices starting at 300
    log_returns = np.random.normal(0.0003, 0.012, n)
    close = 300 * np.exp(np.cumsum(log_returns))

    # Generate OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(50_000_000, 150_000_000, n)

    returns = np.zeros(n)
    returns[1:] = np.diff(np.log(close))

    return pd.DataFrame({
        "date": sample_dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": volume,
        "returns": returns,
    })


@pytest.fixture
def sample_vol_data(sample_dates) -> pd.DataFrame:
    """Generate synthetic VIX-like vol index data."""
    np.random.seed(42)
    n = len(sample_dates)
    vol_level = 20 + np.cumsum(np.random.normal(0, 0.5, n))
    vol_level = np.clip(vol_level, 10, 80)

    return pd.DataFrame({
        "date": sample_dates,
        "vol_level": vol_level,
        "source": "test",
        "quality_flag": "ok",
    })
