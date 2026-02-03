"""Tests for the database layer."""

from datetime import date

import pandas as pd
import pytest

from vol_radar.db.database import Database
from vol_radar.db.models import DimInstrument, DimCalendar, FactPriceDaily
from vol_radar.db.repos import (
    InstrumentRepo,
    CalendarRepo,
    PriceRepo,
    VolIndexRepo,
    FeatureRepo,
    ManifestRepo,
    RegimeRepo,
)


class TestDatabase:
    def test_create_tables(self, db):
        """Tables should be created without error."""
        # Tables are created in the fixture; just verify engine exists
        assert db.engine is not None

    def test_session_context_manager(self, db):
        """Session should commit on normal exit."""
        with db.get_session() as session:
            inst = DimInstrument(
                ticker="TEST", name="Test Instrument",
                asset_class="equity", region="US",
                country="US", currency="USD",
            )
            session.add(inst)

        # Verify persisted
        with db.get_session() as session:
            result = session.query(DimInstrument).filter_by(ticker="TEST").first()
            assert result is not None
            assert result.name == "Test Instrument"

    def test_session_rollback_on_error(self, db):
        """Session should rollback on exception."""
        try:
            with db.get_session() as session:
                inst = DimInstrument(
                    ticker="ROLLBACK_TEST", name="Rollback",
                    asset_class="equity", region="US",
                    country="US", currency="USD",
                )
                session.add(inst)
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify NOT persisted
        with db.get_session() as session:
            result = session.query(DimInstrument).filter_by(ticker="ROLLBACK_TEST").first()
            assert result is None


class TestInstrumentRepo:
    def test_upsert_insert(self, db):
        with db.get_session() as session:
            ids = InstrumentRepo.upsert_instruments(session, [
                {"ticker": "SPY", "name": "SPDR S&P 500", "asset_class": "equity",
                 "region": "US", "country": "US", "currency": "USD",
                 "venue": "NYSE", "timezone": "America/New_York"},
            ])
            assert len(ids) == 1
            assert ids[0] > 0

    def test_upsert_idempotent(self, db):
        with db.get_session() as session:
            ids1 = InstrumentRepo.upsert_instruments(session, [
                {"ticker": "SPY", "name": "SPDR S&P 500", "asset_class": "equity",
                 "region": "US", "country": "US", "currency": "USD",
                 "venue": "NYSE", "timezone": "America/New_York"},
            ])

        with db.get_session() as session:
            ids2 = InstrumentRepo.upsert_instruments(session, [
                {"ticker": "SPY", "name": "SPDR S&P 500 Updated", "asset_class": "equity",
                 "region": "US", "country": "US", "currency": "USD",
                 "venue": "NYSE", "timezone": "America/New_York"},
            ])
            assert ids1[0] == ids2[0]

            # Verify name was updated
            inst = session.query(DimInstrument).filter_by(ticker="SPY").first()
            assert inst.name == "SPDR S&P 500 Updated"

    def test_get_instrument_id(self, db):
        with db.get_session() as session:
            InstrumentRepo.upsert_instruments(session, [
                {"ticker": "QQQ", "name": "QQQ Trust", "asset_class": "equity",
                 "region": "US", "country": "US", "currency": "USD"},
            ])

        with db.get_session() as session:
            iid = InstrumentRepo.get_instrument_id(session, "QQQ")
            assert iid is not None

            missing = InstrumentRepo.get_instrument_id(session, "MISSING")
            assert missing is None

    def test_get_instruments_by_region(self, db):
        with db.get_session() as session:
            InstrumentRepo.upsert_instruments(session, [
                {"ticker": "SPY", "name": "SPY", "asset_class": "equity",
                 "region": "US", "country": "US", "currency": "USD"},
                {"ticker": "VGK", "name": "VGK", "asset_class": "equity",
                 "region": "Europe", "country": "EU", "currency": "USD"},
            ])

        with db.get_session() as session:
            us = InstrumentRepo.get_instruments_by_region(session, "US")
            assert len(us) == 1
            assert us[0].ticker == "SPY"


class TestCalendarRepo:
    def test_ensure_dates(self, db):
        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 3, 29)]
        with db.get_session() as session:
            date_map = CalendarRepo.ensure_dates(session, dates)
            assert len(date_map) == 3
            assert all(isinstance(v, int) for v in date_map.values())

    def test_ensure_dates_idempotent(self, db):
        dates = [date(2024, 1, 2)]
        with db.get_session() as session:
            map1 = CalendarRepo.ensure_dates(session, dates)

        with db.get_session() as session:
            map2 = CalendarRepo.ensure_dates(session, dates)
            assert map1[date(2024, 1, 2)] == map2[date(2024, 1, 2)]

    def test_month_end_detection(self, db):
        dates = [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 15)]
        with db.get_session() as session:
            CalendarRepo.ensure_dates(session, dates)

        with db.get_session() as session:
            jan_end = session.query(DimCalendar).filter_by(date=date(2024, 1, 31)).first()
            assert jan_end.is_month_end is True

            feb_end = session.query(DimCalendar).filter_by(date=date(2024, 2, 29)).first()
            assert feb_end.is_month_end is True

            mid_mar = session.query(DimCalendar).filter_by(date=date(2024, 3, 15)).first()
            assert mid_mar.is_month_end is False

    def test_quarter_end_detection(self, db):
        dates = [date(2024, 3, 31), date(2024, 6, 30), date(2024, 5, 31)]
        with db.get_session() as session:
            CalendarRepo.ensure_dates(session, dates)

        with db.get_session() as session:
            q1_end = session.query(DimCalendar).filter_by(date=date(2024, 3, 31)).first()
            assert q1_end.is_quarter_end is True

            q2_end = session.query(DimCalendar).filter_by(date=date(2024, 6, 30)).first()
            assert q2_end.is_quarter_end is True

            may_end = session.query(DimCalendar).filter_by(date=date(2024, 5, 31)).first()
            assert may_end.is_quarter_end is False


class TestPriceRepo:
    def test_upsert_and_read_prices(self, db, sample_prices):
        with db.get_session() as session:
            ids = InstrumentRepo.upsert_instruments(session, [
                {"ticker": "SPY", "name": "SPY", "asset_class": "equity",
                 "region": "US", "country": "US", "currency": "USD"},
            ])
            inst_id = ids[0]

            dates = [
                d.date() if hasattr(d, "date") else d
                for d in sample_prices["date"].tolist()
            ]
            date_map = CalendarRepo.ensure_dates(session, dates)

            count = PriceRepo.upsert_prices(session, inst_id, sample_prices, date_map)
            assert count == len(sample_prices)

        with db.get_session() as session:
            df = PriceRepo.get_prices(session, inst_id)
            assert len(df) == len(sample_prices)
            assert "close" in df.columns
            assert "returns" in df.columns

    def test_upsert_idempotent(self, db, sample_prices):
        small_df = sample_prices.head(10).copy()

        with db.get_session() as session:
            ids = InstrumentRepo.upsert_instruments(session, [
                {"ticker": "SPY", "name": "SPY", "asset_class": "equity",
                 "region": "US", "country": "US", "currency": "USD"},
            ])
            inst_id = ids[0]
            dates = [d.date() if hasattr(d, "date") else d for d in small_df["date"].tolist()]
            date_map = CalendarRepo.ensure_dates(session, dates)
            PriceRepo.upsert_prices(session, inst_id, small_df, date_map)

        # Upsert again with modified close
        small_df_modified = small_df.copy()
        small_df_modified["close"] = small_df_modified["close"] * 1.01

        with db.get_session() as session:
            dates = [d.date() if hasattr(d, "date") else d for d in small_df_modified["date"].tolist()]
            date_map = CalendarRepo.ensure_dates(session, dates)
            PriceRepo.upsert_prices(session, inst_id, small_df_modified, date_map)

        with db.get_session() as session:
            df = PriceRepo.get_prices(session, inst_id)
            # Should still have 10 rows (upsert, not duplicate)
            assert len(df) == 10


class TestVolIndexRepo:
    def test_upsert_and_read(self, db, sample_vol_data):
        with db.get_session() as session:
            ids = InstrumentRepo.upsert_instruments(session, [
                {"ticker": "^VIX", "name": "VIX", "asset_class": "vol_index",
                 "region": "US", "country": "US", "currency": "USD"},
            ])
            inst_id = ids[0]
            dates = [d.date() if hasattr(d, "date") else d for d in sample_vol_data["date"].tolist()]
            date_map = CalendarRepo.ensure_dates(session, dates)

            count = VolIndexRepo.upsert_vol_data(session, inst_id, sample_vol_data, date_map)
            assert count == len(sample_vol_data)

        with db.get_session() as session:
            df = VolIndexRepo.get_vol_data(session, inst_id)
            assert len(df) == len(sample_vol_data)
            assert "vol_level" in df.columns


class TestManifestRepo:
    def test_insert_and_get(self, db):
        from datetime import datetime

        with db.get_session() as session:
            ManifestRepo.insert_run(session, {
                "run_id": "test_run_001",
                "timestamp": datetime.now(),
                "config_hash": "abc123",
                "status": "running",
            })

        with db.get_session() as session:
            latest = ManifestRepo.get_latest_run(session)
            assert latest is not None
            assert latest.run_id == "test_run_001"
            assert latest.status == "running"

    def test_update_status(self, db):
        from datetime import datetime

        with db.get_session() as session:
            ManifestRepo.insert_run(session, {
                "run_id": "test_run_002",
                "timestamp": datetime.now(),
                "config_hash": "abc123",
                "status": "running",
            })

        with db.get_session() as session:
            ManifestRepo.update_status(session, "test_run_002", "completed")

        with db.get_session() as session:
            latest = ManifestRepo.get_latest_run(session)
            assert latest.status == "completed"
