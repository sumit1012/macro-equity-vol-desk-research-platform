"""Tests for calendar and date alignment."""

from datetime import date

import pandas as pd
import pytest

from vol_radar.db.repos import CalendarRepo


class TestCalendarAlignment:
    def test_weekday_only_dates(self, db):
        """Weekdays should be properly handled."""
        dates = [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]  # Mon, Tue, Wed
        with db.get_session() as session:
            date_map = CalendarRepo.ensure_dates(session, dates)
            assert len(date_map) == 3

    def test_no_interpolation_for_missing_dates(self, db):
        """Only provided dates should exist, no interpolation."""
        dates = [date(2024, 1, 2), date(2024, 1, 5)]  # Tue, Fri (skipping Wed, Thu)
        with db.get_session() as session:
            date_map = CalendarRepo.ensure_dates(session, dates)
            assert len(date_map) == 2
            assert date(2024, 1, 3) not in date_map
            assert date(2024, 1, 4) not in date_map

    def test_month_end_leap_year(self, db):
        """Feb 29 in leap year should be month end."""
        dates = [date(2024, 2, 29)]
        with db.get_session() as session:
            CalendarRepo.ensure_dates(session, dates)
            from vol_radar.db.models import DimCalendar
            entry = session.query(DimCalendar).filter_by(date=date(2024, 2, 29)).first()
            assert entry.is_month_end is True

    def test_quarter_end_only_for_correct_months(self, db):
        """Quarter end should only be for months 3, 6, 9, 12."""
        dates = [
            date(2024, 1, 31),   # Jan - month end but NOT quarter end
            date(2024, 3, 31),   # Mar - quarter end
            date(2024, 4, 30),   # Apr - month end but NOT quarter end
            date(2024, 6, 30),   # Jun - quarter end
            date(2024, 9, 30),   # Sep - quarter end
            date(2024, 12, 31),  # Dec - quarter end
        ]
        with db.get_session() as session:
            CalendarRepo.ensure_dates(session, dates)
            from vol_radar.db.models import DimCalendar

            jan = session.query(DimCalendar).filter_by(date=date(2024, 1, 31)).first()
            assert jan.is_month_end is True
            assert jan.is_quarter_end is False

            mar = session.query(DimCalendar).filter_by(date=date(2024, 3, 31)).first()
            assert mar.is_quarter_end is True

            apr = session.query(DimCalendar).filter_by(date=date(2024, 4, 30)).first()
            assert apr.is_quarter_end is False

            dec = session.query(DimCalendar).filter_by(date=date(2024, 12, 31)).first()
            assert dec.is_quarter_end is True

    def test_iso_week_number(self, db):
        """Week numbers should follow ISO standard."""
        dates = [date(2024, 1, 1)]  # ISO week 1
        with db.get_session() as session:
            CalendarRepo.ensure_dates(session, dates)
            from vol_radar.db.models import DimCalendar
            entry = session.query(DimCalendar).filter_by(date=date(2024, 1, 1)).first()
            assert entry.week == date(2024, 1, 1).isocalendar()[1]
