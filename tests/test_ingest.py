"""Tests for data ingestion layer."""

from __future__ import annotations

import io
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from vol_radar.ingest.base import FallbackClient, MarketDataClient
from vol_radar.ingest.yahoo import YahooClient
from vol_radar.ingest.stooq import StooqClient
from vol_radar.ingest.upload import CsvUploadHandler


class TestMarketDataClientValidation:
    """Test base validation methods."""

    def test_validate_ohlcv_valid(self):
        client = YahooClient()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=5),
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "adj_close": [102, 103, 104, 105, 106],
            "volume": [1000, 2000, 3000, 4000, 5000],
        })
        result = client.validate_ohlcv(df, "TEST")
        assert len(result) == 5
        assert "returns" in result.columns

    def test_validate_ohlcv_missing_column(self):
        client = YahooClient()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3),
            "open": [100, 101, 102],
            # missing high, low, close
        })
        with pytest.raises(ValueError, match="Missing required column"):
            client.validate_ohlcv(df, "TEST")

    def test_validate_ohlcv_adds_adj_close(self):
        client = YahooClient()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3),
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [102, 103, 104],
        })
        result = client.validate_ohlcv(df, "TEST")
        assert "adj_close" in result.columns
        assert result["adj_close"].equals(result["close"])

    def test_validate_ohlcv_empty(self):
        client = YahooClient()
        df = pd.DataFrame()
        result = client.validate_ohlcv(df, "TEST")
        assert result.empty

    def test_validate_vol_index_valid(self):
        client = YahooClient()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3),
            "vol_level": [15.0, 16.0, 14.5],
            "source": "test",
            "quality_flag": "ok",
        })
        result = client.validate_vol_index(df, "TEST")
        assert len(result) == 3

    def test_validate_vol_index_drops_negative(self):
        client = YahooClient()
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3),
            "vol_level": [15.0, -1.0, 14.5],
            "source": "test",
            "quality_flag": "ok",
        })
        result = client.validate_vol_index(df, "TEST")
        assert len(result) == 2


class TestFallbackClient:
    def test_fallback_uses_first_success(self):
        mock_client1 = MagicMock(spec=MarketDataClient)
        mock_client1.fetch_ohlcv.return_value = pd.DataFrame({
            "date": [date(2024, 1, 2)],
            "open": [100], "high": [105], "low": [95],
            "close": [102], "adj_close": [102], "volume": [1000],
            "returns": [0.01],
        })

        mock_client2 = MagicMock(spec=MarketDataClient)

        client = FallbackClient([mock_client1, mock_client2])
        result = client.fetch_ohlcv("SPY", date(2024, 1, 2), date(2024, 1, 5))

        assert len(result) == 1
        mock_client2.fetch_ohlcv.assert_not_called()

    def test_fallback_tries_second_on_failure(self):
        mock_client1 = MagicMock(spec=MarketDataClient)
        mock_client1.__class__.__name__ = "Client1"
        mock_client1.fetch_ohlcv.side_effect = Exception("API error")

        mock_client2 = MagicMock(spec=MarketDataClient)
        mock_client2.__class__.__name__ = "Client2"
        mock_client2.fetch_ohlcv.return_value = pd.DataFrame({
            "date": [date(2024, 1, 2)],
            "open": [100], "high": [105], "low": [95],
            "close": [102], "adj_close": [102], "volume": [1000],
            "returns": [0.01],
        })

        client = FallbackClient([mock_client1, mock_client2])
        result = client.fetch_ohlcv("SPY", date(2024, 1, 2), date(2024, 1, 5))

        assert len(result) == 1

    def test_fallback_returns_empty_when_all_fail(self):
        mock_client1 = MagicMock(spec=MarketDataClient)
        mock_client1.__class__.__name__ = "Client1"
        mock_client1.fetch_ohlcv.side_effect = Exception("Error1")

        mock_client2 = MagicMock(spec=MarketDataClient)
        mock_client2.__class__.__name__ = "Client2"
        mock_client2.fetch_ohlcv.side_effect = Exception("Error2")

        client = FallbackClient([mock_client1, mock_client2])
        result = client.fetch_ohlcv("SPY", date(2024, 1, 2), date(2024, 1, 5))

        assert result.empty

    def test_fallback_requires_clients(self):
        with pytest.raises(ValueError, match="At least one client"):
            FallbackClient([])


class TestCsvUploadHandler:
    def test_parse_valid_csv(self, tmp_path):
        csv_content = """date,ticker,vol_level,source,quality_flag
2024-01-02,^VSTOXX,15.5,manual,ok
2024-01-03,^VSTOXX,16.2,manual,ok
2024-01-04,^VSTOXX,14.8,manual,ok"""

        csv_file = tmp_path / "test_vol.csv"
        csv_file.write_text(csv_content)

        handler = CsvUploadHandler()
        df = handler.parse_vol_index_csv(str(csv_file))

        assert len(df) == 3
        assert set(df.columns) >= {"date", "ticker", "vol_level", "source", "quality_flag"}
        assert df["ticker"].unique().tolist() == ["^VSTOXX"]

    def test_parse_missing_columns(self, tmp_path):
        csv_content = """date,vol_level
2024-01-02,15.5"""

        csv_file = tmp_path / "bad_vol.csv"
        csv_file.write_text(csv_content)

        handler = CsvUploadHandler()
        with pytest.raises(ValueError, match="missing required columns"):
            handler.parse_vol_index_csv(str(csv_file))

    def test_parse_drops_invalid_rows(self, tmp_path):
        csv_content = """date,ticker,vol_level,source,quality_flag
2024-01-02,^VSTOXX,15.5,manual,ok
2024-01-03,^VSTOXX,-1.0,manual,ok
2024-01-04,^VSTOXX,,manual,ok
2024-01-05,^VSTOXX,14.8,manual,ok"""

        csv_file = tmp_path / "partial_vol.csv"
        csv_file.write_text(csv_content)

        handler = CsvUploadHandler()
        df = handler.parse_vol_index_csv(str(csv_file))

        # -1.0 and NaN should be dropped
        assert len(df) == 2

    def test_file_not_found(self):
        handler = CsvUploadHandler()
        with pytest.raises(FileNotFoundError):
            handler.parse_vol_index_csv("/nonexistent/file.csv")

    def test_store_uploaded_data(self, db):
        """Test storing uploaded data into the database."""
        from vol_radar.db.repos import InstrumentRepo, CalendarRepo

        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["^VSTOXX", "^VSTOXX"],
            "vol_level": [15.5, 16.2],
            "source": ["manual", "manual"],
            "quality_flag": ["ok", "ok"],
        })

        handler = CsvUploadHandler()

        with db.get_session() as session:
            count = handler.store_uploaded_data(df, session)
            assert count == 2
