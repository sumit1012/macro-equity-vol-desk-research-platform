"""Manual CSV upload handler for vol index data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from vol_radar.db.repos import CalendarRepo, InstrumentRepo, VolIndexRepo


class CsvUploadHandler:
    """Handles manual upload of volatility index data via CSV files.

    Expected CSV format:
        date,ticker,vol_level,source,quality_flag
        2024-01-02,^VSTOXX,15.5,manual,ok
        2024-01-03,^VSTOXX,16.2,manual,ok
    """

    REQUIRED_COLUMNS = ["date", "ticker", "vol_level", "source", "quality_flag"]

    def parse_vol_index_csv(self, file_path: str | Path) -> pd.DataFrame:
        """Parse a vol index CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Validated DataFrame with columns: date, ticker, vol_level, source, quality_flag

        Raises:
            ValueError: If the CSV is malformed or missing required columns.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        logger.info(f"Parsing vol index CSV: {path}")
        df = pd.read_csv(path)

        # Check required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Expected columns: {self.REQUIRED_COLUMNS}"
            )

        # Validate and convert types
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df["vol_level"] = pd.to_numeric(df["vol_level"], errors="coerce")
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df["source"] = df["source"].astype(str).str.strip()
        df["quality_flag"] = df["quality_flag"].astype(str).str.strip()

        # Drop invalid rows
        initial_len = len(df)
        df = df.dropna(subset=["date", "ticker", "vol_level"])
        df = df[df["vol_level"] >= 0]
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} invalid rows from CSV")

        # Validate no empty tickers
        if (df["ticker"] == "").any():
            raise ValueError("CSV contains rows with empty ticker values")

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        logger.info(f"Parsed {len(df)} valid rows from CSV")
        return df

    def store_uploaded_data(self, df: pd.DataFrame, session) -> int:
        """Store parsed vol index data into the database.

        Args:
            df: Validated DataFrame from parse_vol_index_csv.
            session: SQLAlchemy session.

        Returns:
            Number of rows stored.
        """
        total_stored = 0

        # Get instrument mapping
        instrument_map = InstrumentRepo.get_instrument_map(session)

        # Ensure all dates exist
        dates = [d.date() if hasattr(d, "date") else d for d in df["date"].unique()]
        date_id_map = CalendarRepo.ensure_dates(session, dates)

        # Process each ticker
        for ticker in df["ticker"].unique():
            instrument_id = instrument_map.get(ticker)
            if instrument_id is None:
                # Auto-create the instrument as a vol_index type
                ids = InstrumentRepo.upsert_instruments(session, [{
                    "ticker": ticker,
                    "name": f"Vol Index {ticker}",
                    "asset_class": "vol_index",
                    "region": "unknown",
                    "country": "unknown",
                    "currency": "USD",
                }])
                instrument_id = ids[0]
                logger.info(f"Auto-created instrument for uploaded ticker: {ticker}")

            ticker_df = df[df["ticker"] == ticker].copy()
            count = VolIndexRepo.upsert_vol_data(session, instrument_id, ticker_df, date_id_map)
            total_stored += count

        logger.info(f"Stored {total_stored} vol index rows from CSV upload")
        return total_stored

    @staticmethod
    def parse_from_streamlit_upload(uploaded_file) -> pd.DataFrame:
        """Parse a Streamlit UploadedFile object.

        Args:
            uploaded_file: Streamlit UploadedFile object.

        Returns:
            Validated DataFrame.
        """
        df = pd.read_csv(uploaded_file)

        required = CsvUploadHandler.REQUIRED_COLUMNS
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Expected columns: {required}"
            )

        df["date"] = pd.to_datetime(df["date"], format="mixed")
        df["vol_level"] = pd.to_numeric(df["vol_level"], errors="coerce")
        df = df.dropna(subset=["date", "ticker", "vol_level"])
        df = df[df["vol_level"] >= 0]

        return df.sort_values(["ticker", "date"]).reset_index(drop=True)
