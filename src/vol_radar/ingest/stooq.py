"""Stooq data client as fallback source."""

from __future__ import annotations

import io
import time
from datetime import date

import pandas as pd
import requests
from loguru import logger

from vol_radar.ingest.base import MarketDataClient


class StooqClient(MarketDataClient):
    """Fetches market data from Stooq.com via CSV downloads."""

    BASE_URL = "https://stooq.com/q/d/l/"
    TICKER_MAP = {
        # US ETFs
        "SPY": "spy.us",
        "QQQ": "qqq.us",
        "IWM": "iwm.us",
        # Europe ETFs
        "VGK": "vgk.us",
        "FEZ": "fez.us",
        "EWU": "ewu.us",
        "EWG": "ewg.us",
        "EWQ": "ewq.us",
        # Asia ETFs
        "EWJ": "ewj.us",
        "FXI": "fxi.us",
        "EWH": "ewh.us",
        "EWY": "ewy.us",
        "INDA": "inda.us",
        # FX
        "UUP": "uup.us",
        "FXE": "fxe.us",
        "FXY": "fxy.us",
    }

    def __init__(self, rate_limit_seconds: float = 1.0):
        self._rate_limit = rate_limit_seconds

    def _stooq_ticker(self, ticker: str) -> str:
        """Convert standard ticker to Stooq format."""
        return self.TICKER_MAP.get(ticker, f"{ticker.lower()}.us")

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch OHLCV data from Stooq."""
        stooq_ticker = self._stooq_ticker(ticker)
        d1 = start.strftime("%Y%m%d")
        d2 = end.strftime("%Y%m%d")

        url = f"{self.BASE_URL}?s={stooq_ticker}&d1={d1}&d2={d2}&i=d"
        logger.debug(f"Fetching OHLCV from Stooq for {ticker}: {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))

            if df.empty or len(df) < 2:
                logger.warning(f"No data from Stooq for {ticker}")
                return pd.DataFrame(columns=self.OHLCV_COLUMNS + ["returns"])

            # Stooq uses capitalized column names
            col_map = {
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
            df = df.rename(columns=col_map)

            # Add adj_close (Stooq doesn't provide adjusted prices)
            df["adj_close"] = df["close"]

            df = self.validate_ohlcv(df, ticker)
            time.sleep(self._rate_limit)
            return df

        except Exception as e:
            logger.error(f"Stooq error for {ticker}: {e}")
            raise

    def fetch_vol_index(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Stooq generally doesn't have vol indices. Returns empty."""
        logger.debug(f"Stooq does not support vol indices, skipping {ticker}")
        return pd.DataFrame(columns=self.VOL_INDEX_COLUMNS)
