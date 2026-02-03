"""Yahoo Finance data client using yfinance."""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from vol_radar.ingest.base import MarketDataClient


class YahooClient(MarketDataClient):
    """Fetches market data from Yahoo Finance via yfinance."""

    def __init__(self, rate_limit_seconds: float = 0.5):
        self._rate_limit = rate_limit_seconds

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        import yfinance as yf

        logger.debug(f"Fetching OHLCV from Yahoo for {ticker}: {start} to {end}")

        try:
            data = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=False,
            )

            if data.empty:
                logger.warning(f"No data returned from Yahoo for {ticker}")
                return pd.DataFrame(columns=self.OHLCV_COLUMNS + ["returns"])

            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = pd.DataFrame({
                "date": data.index,
                "open": data["Open"].values,
                "high": data["High"].values,
                "low": data["Low"].values,
                "close": data["Close"].values,
                "adj_close": data["Adj Close"].values if "Adj Close" in data.columns else data["Close"].values,
                "volume": data["Volume"].values,
            })

            df = self.validate_ohlcv(df, ticker)
            time.sleep(self._rate_limit)
            return df

        except Exception as e:
            logger.error(f"Yahoo Finance error for {ticker}: {e}")
            raise

    def fetch_vol_index(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch vol index data from Yahoo Finance (works for ^VIX, ^VIX9D, ^VIX3M)."""
        import yfinance as yf

        logger.debug(f"Fetching vol index from Yahoo for {ticker}: {start} to {end}")

        try:
            data = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=False,
            )

            if data.empty:
                logger.warning(f"No vol index data from Yahoo for {ticker}")
                return pd.DataFrame(columns=self.VOL_INDEX_COLUMNS)

            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = pd.DataFrame({
                "date": data.index,
                "vol_level": data["Close"].values,
                "source": "yahoo",
                "quality_flag": "direct",
            })

            df = self.validate_vol_index(df, ticker)
            time.sleep(self._rate_limit)
            return df

        except Exception as e:
            logger.error(f"Yahoo Finance vol index error for {ticker}: {e}")
            raise
