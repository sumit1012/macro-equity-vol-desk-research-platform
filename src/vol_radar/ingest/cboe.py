"""CBOE VIX futures data client."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests
from loguru import logger


class CboeClient:
    """Fetches VIX futures data from CBOE free CSV downloads."""

    VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    VIX_FUTURES_BASE = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/"

    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    def fetch_vix_history(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Fetch historical VIX index data from CBOE.

        Returns:
            DataFrame with columns: date, vol_level, source, quality_flag
        """
        logger.debug("Fetching VIX history from CBOE")

        try:
            response = requests.get(self.VIX_HISTORY_URL, timeout=self._timeout)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))

            # CBOE CSV has columns: DATE, OPEN, HIGH, LOW, CLOSE
            col_map = {
                "DATE": "date",
                "CLOSE": "vol_level",
            }

            # Try different column name variants
            for orig, target in list(col_map.items()):
                if orig not in df.columns:
                    # Try lowercase
                    lower = orig.lower()
                    if lower in df.columns:
                        col_map[lower] = target
                        del col_map[orig]
                    # Try title case
                    elif orig.title() in df.columns:
                        col_map[orig.title()] = target
                        del col_map[orig]

            df = df.rename(columns=col_map)

            if "date" not in df.columns or "vol_level" not in df.columns:
                logger.error(f"Unexpected CBOE CSV columns: {df.columns.tolist()}")
                return pd.DataFrame(columns=["date", "vol_level", "source", "quality_flag"])

            df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
            df["vol_level"] = pd.to_numeric(df["vol_level"], errors="coerce")
            df["source"] = "cboe"
            df["quality_flag"] = "direct"

            df = df[["date", "vol_level", "source", "quality_flag"]].dropna(
                subset=["vol_level"]
            )

            # Filter by date range
            if start:
                df = df[df["date"] >= pd.Timestamp(start)]
            if end:
                df = df[df["date"] <= pd.Timestamp(end)]

            df = df.sort_values("date").reset_index(drop=True)
            logger.info(f"Fetched {len(df)} VIX history rows from CBOE")
            return df

        except Exception as e:
            logger.error(f"CBOE VIX history fetch error: {e}")
            return pd.DataFrame(columns=["date", "vol_level", "source", "quality_flag"])

    def fetch_vix_futures_term_structure(self) -> pd.DataFrame:
        """Fetch VIX futures term structure data.

        Attempts to download VIX futures settlement prices for constructing
        the term structure curve.

        Returns:
            DataFrame with columns: date, contract_month, settle_price, volume, open_interest
        """
        logger.debug("Fetching VIX futures term structure from CBOE")

        # CBOE provides futures data in various formats
        # We'll try the main settlement URL
        urls_to_try = [
            f"{self.VIX_FUTURES_BASE}VX+VIX_Futures/VX_History.csv",
            "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX+VIX-Futures/VX_History.csv",
        ]

        for url in urls_to_try:
            try:
                response = requests.get(url, timeout=self._timeout)
                if response.status_code != 200:
                    continue

                df = pd.read_csv(io.StringIO(response.text))

                if df.empty:
                    continue

                # Normalize column names
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

                # Try to identify key columns
                date_col = None
                for candidate in ["trade_date", "date", "trade date"]:
                    if candidate in df.columns:
                        date_col = candidate
                        break

                if date_col is None:
                    logger.warning(f"Cannot identify date column in CBOE futures: {df.columns.tolist()}")
                    continue

                settle_col = None
                for candidate in ["settle", "settlement_price", "close", "settle_price"]:
                    if candidate in df.columns:
                        settle_col = candidate
                        break

                if settle_col is None:
                    logger.warning(f"Cannot identify settle column in CBOE futures: {df.columns.tolist()}")
                    continue

                # Build standardized DataFrame
                result = pd.DataFrame({
                    "date": pd.to_datetime(df[date_col], format="mixed"),
                    "settle_price": pd.to_numeric(df[settle_col], errors="coerce"),
                })

                # Try to get contract month
                for candidate in ["futures", "contract", "expiration_date", "contract_month"]:
                    if candidate in df.columns:
                        result["contract_month"] = df[candidate]
                        break

                if "contract_month" not in result.columns:
                    result["contract_month"] = "unknown"

                # Try to get volume and OI
                for col_name, candidates in [
                    ("volume", ["total_volume", "volume", "vol"]),
                    ("open_interest", ["efp", "open_interest", "oi"]),
                ]:
                    for candidate in candidates:
                        if candidate in df.columns:
                            result[col_name] = pd.to_numeric(df[candidate], errors="coerce")
                            break
                    if col_name not in result.columns:
                        result[col_name] = 0

                result = result.dropna(subset=["settle_price"])
                result = result.sort_values("date").reset_index(drop=True)

                logger.info(f"Fetched {len(result)} VIX futures rows from CBOE")
                return result

            except Exception as e:
                logger.warning(f"CBOE futures URL failed ({url}): {e}")
                continue

        logger.warning("Could not fetch VIX futures term structure from any CBOE URL")
        return pd.DataFrame(
            columns=["date", "contract_month", "settle_price", "volume", "open_interest"]
        )

    def get_term_structure_snapshot(self, target_date: date) -> pd.DataFrame:
        """Get VIX futures term structure for a specific date.

        Returns sorted futures prices by contract expiry for the given date.
        """
        full_data = self.fetch_vix_futures_term_structure()
        if full_data.empty:
            return full_data

        snapshot = full_data[full_data["date"].dt.date == target_date].copy()
        if snapshot.empty:
            # Try nearest date
            full_data["date_only"] = full_data["date"].dt.date
            nearest_date = min(
                full_data["date_only"].unique(),
                key=lambda d: abs((d - target_date).days),
            )
            snapshot = full_data[full_data["date_only"] == nearest_date].copy()
            snapshot = snapshot.drop(columns=["date_only"])

        return snapshot.sort_values("contract_month").reset_index(drop=True)
