"""Base classes for market data ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
from loguru import logger


class MarketDataClient(ABC):
    """Abstract base class for market data clients."""

    OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    VOL_INDEX_COLUMNS = ["date", "vol_level", "source", "quality_flag"]

    @abstractmethod
    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker.

        Returns:
            DataFrame with columns: date, open, high, low, close, adj_close, volume
        """
        ...

    @abstractmethod
    def fetch_vol_index(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch volatility index data.

        Returns:
            DataFrame with columns: date, vol_level, source, quality_flag
        """
        ...

    def validate_ohlcv(self, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """Validate and clean OHLCV data."""
        if df.empty:
            logger.warning(f"Empty OHLCV data for {ticker}")
            return df

        # Ensure required columns
        for col in ["date", "open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add adj_close if missing
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        # Add volume if missing
        if "volume" not in df.columns:
            df["volume"] = 0

        # Convert date column
        df["date"] = pd.to_datetime(df["date"])

        # Drop rows with negative prices
        price_cols = ["open", "high", "low", "close", "adj_close"]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        mask = (df[price_cols] > 0).all(axis=1) | df[price_cols].isna().all(axis=1)
        dropped = (~mask).sum()
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with negative prices for {ticker}")
            df = df[mask].copy()

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Compute returns
        df["returns"] = df["adj_close"].pct_change().apply(
            lambda x: float(x) if pd.notna(x) else None
        )

        return df

    def validate_vol_index(self, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """Validate and clean vol index data."""
        if df.empty:
            logger.warning(f"Empty vol index data for {ticker}")
            return df

        if "date" not in df.columns or "vol_level" not in df.columns:
            raise ValueError("Vol index data must have 'date' and 'vol_level' columns")

        df["date"] = pd.to_datetime(df["date"])
        df["vol_level"] = pd.to_numeric(df["vol_level"], errors="coerce")

        if "source" not in df.columns:
            df["source"] = "unknown"
        if "quality_flag" not in df.columns:
            df["quality_flag"] = "ok"

        # Drop rows where vol_level is NaN or negative
        valid = df["vol_level"].notna() & (df["vol_level"] >= 0)
        dropped = (~valid).sum()
        if dropped > 0:
            logger.warning(f"Dropped {dropped} invalid vol index rows for {ticker}")
            df = df[valid].copy()

        df = df.sort_values("date").reset_index(drop=True)
        return df


class FallbackClient:
    """Tries multiple MarketDataClient instances in order until one succeeds."""

    def __init__(self, clients: list[MarketDataClient]):
        if not clients:
            raise ValueError("At least one client is required")
        self._clients = clients

    def fetch_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Try each client until OHLCV data is fetched."""
        for client in self._clients:
            try:
                df = client.fetch_ohlcv(ticker, start, end)
                if not df.empty:
                    logger.info(
                        f"Fetched {len(df)} OHLCV rows for {ticker} "
                        f"from {client.__class__.__name__}"
                    )
                    return df
            except Exception as e:
                logger.warning(
                    f"{client.__class__.__name__} failed for {ticker} OHLCV: {e}"
                )
                continue

        logger.error(f"All clients failed to fetch OHLCV for {ticker}")
        return pd.DataFrame(columns=MarketDataClient.OHLCV_COLUMNS + ["returns"])

    def fetch_vol_index(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Try each client until vol index data is fetched."""
        for client in self._clients:
            try:
                df = client.fetch_vol_index(ticker, start, end)
                if not df.empty:
                    logger.info(
                        f"Fetched {len(df)} vol index rows for {ticker} "
                        f"from {client.__class__.__name__}"
                    )
                    return df
            except Exception as e:
                logger.warning(
                    f"{client.__class__.__name__} failed for {ticker} vol index: {e}"
                )
                continue

        logger.warning(f"No vol index data available for {ticker} from any client")
        return pd.DataFrame(columns=MarketDataClient.VOL_INDEX_COLUMNS)
