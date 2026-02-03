"""Repository classes for database operations."""

from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from vol_radar.db.models import (
    DimCalendar,
    DimInstrument,
    DimRegime,
    FactBacktestPerf,
    FactBacktestTrades,
    FactDislocationEvents,
    FactFeaturesDaily,
    FactPriceDaily,
    FactVolIndexDaily,
    RunManifest,
)


class InstrumentRepo:
    """Repository for dim_instrument operations."""

    @staticmethod
    def upsert_instruments(session: Session, instruments: list[dict]) -> list[int]:
        """Insert or update instruments. Returns list of instrument_ids."""
        ids = []
        for inst in instruments:
            existing = session.execute(
                select(DimInstrument).where(DimInstrument.ticker == inst["ticker"])
            ).scalar_one_or_none()

            if existing:
                for key, value in inst.items():
                    if key != "ticker":
                        setattr(existing, key, value)
                ids.append(existing.instrument_id)
            else:
                obj = DimInstrument(**inst)
                session.add(obj)
                session.flush()
                ids.append(obj.instrument_id)

        logger.debug(f"Upserted {len(instruments)} instruments")
        return ids

    @staticmethod
    def get_instrument_id(session: Session, ticker: str) -> int | None:
        """Get instrument_id for a ticker."""
        result = session.execute(
            select(DimInstrument.instrument_id).where(DimInstrument.ticker == ticker)
        ).scalar_one_or_none()
        return result

    @staticmethod
    def get_all_instruments(session: Session) -> list[DimInstrument]:
        """Get all instruments."""
        return list(session.execute(select(DimInstrument)).scalars().all())

    @staticmethod
    def get_instruments_by_region(session: Session, region: str) -> list[DimInstrument]:
        """Get instruments filtered by region."""
        return list(
            session.execute(
                select(DimInstrument).where(DimInstrument.region == region)
            ).scalars().all()
        )

    @staticmethod
    def get_instrument_map(session: Session) -> dict[str, int]:
        """Get ticker -> instrument_id mapping."""
        instruments = session.execute(
            select(DimInstrument.ticker, DimInstrument.instrument_id)
        ).all()
        return {ticker: iid for ticker, iid in instruments}


class CalendarRepo:
    """Repository for dim_calendar operations."""

    @staticmethod
    def ensure_dates(session: Session, dates: list[date]) -> dict[date, int]:
        """Ensure all dates exist in dim_calendar. Returns date -> date_id mapping."""
        # Get existing dates
        existing = session.execute(
            select(DimCalendar.date, DimCalendar.date_id)
        ).all()
        existing_map = {d: did for d, did in existing}

        date_id_map = {}
        for d in dates:
            if d in existing_map:
                date_id_map[d] = existing_map[d]
            else:
                # Compute calendar attributes
                last_day = calendar.monthrange(d.year, d.month)[1]
                is_month_end = d.day == last_day
                is_quarter_end = is_month_end and d.month in (3, 6, 9, 12)

                obj = DimCalendar(
                    date=d,
                    year=d.year,
                    month=d.month,
                    week=d.isocalendar()[1],
                    is_month_end=is_month_end,
                    is_quarter_end=is_quarter_end,
                )
                session.add(obj)
                session.flush()
                date_id_map[d] = obj.date_id
                existing_map[d] = obj.date_id

        logger.debug(f"Ensured {len(dates)} dates in calendar")
        return date_id_map

    @staticmethod
    def get_date_id(session: Session, d: date) -> int | None:
        """Get date_id for a specific date."""
        return session.execute(
            select(DimCalendar.date_id).where(DimCalendar.date == d)
        ).scalar_one_or_none()

    @staticmethod
    def get_date_map(session: Session) -> dict[date, int]:
        """Get full date -> date_id mapping."""
        results = session.execute(select(DimCalendar.date, DimCalendar.date_id)).all()
        return {d: did for d, did in results}

    @staticmethod
    def get_date_range(session: Session, start: date, end: date) -> list[DimCalendar]:
        """Get calendar entries within a date range."""
        return list(
            session.execute(
                select(DimCalendar)
                .where(DimCalendar.date >= start, DimCalendar.date <= end)
                .order_by(DimCalendar.date)
            ).scalars().all()
        )


class PriceRepo:
    """Repository for fact_price_daily operations."""

    @staticmethod
    def upsert_prices(
        session: Session, instrument_id: int, df: pd.DataFrame, date_id_map: dict[date, int]
    ) -> int:
        """Upsert daily price data from a DataFrame.

        Args:
            session: DB session
            instrument_id: ID of the instrument
            df: DataFrame with columns [date, open, high, low, close, adj_close, volume, returns]
            date_id_map: Mapping from date to date_id

        Returns:
            Number of rows upserted.
        """
        count = 0
        for _, row in df.iterrows():
            d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            date_id = date_id_map.get(d)
            if date_id is None:
                continue

            existing = session.execute(
                select(FactPriceDaily).where(
                    FactPriceDaily.instrument_id == instrument_id,
                    FactPriceDaily.date_id == date_id,
                )
            ).scalar_one_or_none()

            if existing:
                existing.open = float(row.get("open", 0) or 0)
                existing.high = float(row.get("high", 0) or 0)
                existing.low = float(row.get("low", 0) or 0)
                existing.close = float(row.get("close", 0) or 0)
                existing.adj_close = float(row.get("adj_close", 0) or 0)
                existing.volume = int(row.get("volume", 0) or 0)
                existing.returns = float(row["returns"]) if pd.notna(row.get("returns")) else None
            else:
                obj = FactPriceDaily(
                    instrument_id=instrument_id,
                    date_id=date_id,
                    open=float(row.get("open", 0) or 0),
                    high=float(row.get("high", 0) or 0),
                    low=float(row.get("low", 0) or 0),
                    close=float(row.get("close", 0) or 0),
                    adj_close=float(row.get("adj_close", 0) or 0),
                    volume=int(row.get("volume", 0) or 0),
                    returns=float(row["returns"]) if pd.notna(row.get("returns")) else None,
                )
                session.add(obj)
            count += 1

        logger.debug(f"Upserted {count} price rows for instrument {instrument_id}")
        return count

    @staticmethod
    def get_prices(
        session: Session,
        instrument_id: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Get price data as a DataFrame."""
        query = (
            select(
                DimCalendar.date,
                FactPriceDaily.open,
                FactPriceDaily.high,
                FactPriceDaily.low,
                FactPriceDaily.close,
                FactPriceDaily.adj_close,
                FactPriceDaily.volume,
                FactPriceDaily.returns,
            )
            .join(DimCalendar, DimCalendar.date_id == FactPriceDaily.date_id)
            .where(FactPriceDaily.instrument_id == instrument_id)
        )
        if start_date:
            query = query.where(DimCalendar.date >= start_date)
        if end_date:
            query = query.where(DimCalendar.date <= end_date)
        query = query.order_by(DimCalendar.date)

        result = session.execute(query).all()
        if not result:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "adj_close", "volume", "returns"]
            )

        df = pd.DataFrame(result, columns=["date", "open", "high", "low", "close", "adj_close", "volume", "returns"])
        return df


class VolIndexRepo:
    """Repository for fact_vol_index_daily operations."""

    @staticmethod
    def upsert_vol_data(
        session: Session, instrument_id: int, df: pd.DataFrame, date_id_map: dict[date, int]
    ) -> int:
        """Upsert vol index data."""
        count = 0
        for _, row in df.iterrows():
            d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            date_id = date_id_map.get(d)
            if date_id is None:
                continue

            existing = session.execute(
                select(FactVolIndexDaily).where(
                    FactVolIndexDaily.instrument_id == instrument_id,
                    FactVolIndexDaily.date_id == date_id,
                )
            ).scalar_one_or_none()

            vol_level = float(row["vol_level"]) if pd.notna(row.get("vol_level")) else None
            source = str(row.get("source", "unknown"))
            quality_flag = str(row.get("quality_flag", "ok"))

            if existing:
                existing.vol_level = vol_level
                existing.source = source
                existing.quality_flag = quality_flag
            else:
                obj = FactVolIndexDaily(
                    instrument_id=instrument_id,
                    date_id=date_id,
                    vol_level=vol_level,
                    source=source,
                    quality_flag=quality_flag,
                )
                session.add(obj)
            count += 1

        logger.debug(f"Upserted {count} vol index rows for instrument {instrument_id}")
        return count

    @staticmethod
    def get_vol_data(
        session: Session,
        instrument_id: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Get vol index data as a DataFrame."""
        query = (
            select(
                DimCalendar.date,
                FactVolIndexDaily.vol_level,
                FactVolIndexDaily.source,
                FactVolIndexDaily.quality_flag,
            )
            .join(DimCalendar, DimCalendar.date_id == FactVolIndexDaily.date_id)
            .where(FactVolIndexDaily.instrument_id == instrument_id)
        )
        if start_date:
            query = query.where(DimCalendar.date >= start_date)
        if end_date:
            query = query.where(DimCalendar.date <= end_date)
        query = query.order_by(DimCalendar.date)

        result = session.execute(query).all()
        if not result:
            return pd.DataFrame(columns=["date", "vol_level", "source", "quality_flag"])

        return pd.DataFrame(result, columns=["date", "vol_level", "source", "quality_flag"])


class FeatureRepo:
    """Repository for fact_features_daily operations."""

    @staticmethod
    def upsert_features(
        session: Session, instrument_id: int, df: pd.DataFrame, date_id_map: dict[date, int]
    ) -> int:
        """Upsert feature data from a DataFrame."""
        feature_cols = [
            "rv_5d", "rv_21d", "rv_63d", "vrp_5d", "vrp_21d", "vrp_63d",
            "curve_slope", "curve_curvature", "roll_down_proxy",
            "skew_proxy", "vol_of_vol", "drawdown",
            "flow_dollar_volume_zscore", "flow_impact_proxy", "div_shock_flag",
            "regime_score", "z_vrp_21d", "z_curve_slope", "z_skew_proxy",
            "composite_dislocation_score",
        ]

        count = 0
        for _, row in df.iterrows():
            d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            date_id = date_id_map.get(d)
            if date_id is None:
                continue

            existing = session.execute(
                select(FactFeaturesDaily).where(
                    FactFeaturesDaily.instrument_id == instrument_id,
                    FactFeaturesDaily.date_id == date_id,
                )
            ).scalar_one_or_none()

            data = {}
            for col in feature_cols:
                if col in row.index:
                    val = row[col]
                    if col == "div_shock_flag":
                        data[col] = bool(val) if pd.notna(val) else False
                    else:
                        data[col] = float(val) if pd.notna(val) else None
                else:
                    data[col] = None

            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
            else:
                obj = FactFeaturesDaily(
                    instrument_id=instrument_id,
                    date_id=date_id,
                    **data,
                )
                session.add(obj)
            count += 1

        logger.debug(f"Upserted {count} feature rows for instrument {instrument_id}")
        return count

    @staticmethod
    def get_features(
        session: Session,
        instrument_ids: list[int] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Get features as a DataFrame."""
        query = (
            select(
                FactFeaturesDaily.instrument_id,
                DimCalendar.date,
                FactFeaturesDaily.rv_5d,
                FactFeaturesDaily.rv_21d,
                FactFeaturesDaily.rv_63d,
                FactFeaturesDaily.vrp_5d,
                FactFeaturesDaily.vrp_21d,
                FactFeaturesDaily.vrp_63d,
                FactFeaturesDaily.curve_slope,
                FactFeaturesDaily.curve_curvature,
                FactFeaturesDaily.roll_down_proxy,
                FactFeaturesDaily.skew_proxy,
                FactFeaturesDaily.vol_of_vol,
                FactFeaturesDaily.drawdown,
                FactFeaturesDaily.flow_dollar_volume_zscore,
                FactFeaturesDaily.flow_impact_proxy,
                FactFeaturesDaily.div_shock_flag,
                FactFeaturesDaily.regime_score,
                FactFeaturesDaily.z_vrp_21d,
                FactFeaturesDaily.z_curve_slope,
                FactFeaturesDaily.z_skew_proxy,
                FactFeaturesDaily.composite_dislocation_score,
            )
            .join(DimCalendar, DimCalendar.date_id == FactFeaturesDaily.date_id)
        )

        if instrument_ids:
            query = query.where(FactFeaturesDaily.instrument_id.in_(instrument_ids))
        if start_date:
            query = query.where(DimCalendar.date >= start_date)
        if end_date:
            query = query.where(DimCalendar.date <= end_date)
        query = query.order_by(DimCalendar.date)

        result = session.execute(query).all()
        cols = [
            "instrument_id", "date", "rv_5d", "rv_21d", "rv_63d",
            "vrp_5d", "vrp_21d", "vrp_63d",
            "curve_slope", "curve_curvature", "roll_down_proxy",
            "skew_proxy", "vol_of_vol", "drawdown",
            "flow_dollar_volume_zscore", "flow_impact_proxy", "div_shock_flag",
            "regime_score", "z_vrp_21d", "z_curve_slope", "z_skew_proxy",
            "composite_dislocation_score",
        ]
        if not result:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(result, columns=cols)

    @staticmethod
    def get_latest_features(session: Session) -> pd.DataFrame:
        """Get the most recent features for all instruments."""
        # Get the max date_id
        max_date_id = session.execute(
            select(FactFeaturesDaily.date_id).order_by(FactFeaturesDaily.date_id.desc()).limit(1)
        ).scalar_one_or_none()

        if max_date_id is None:
            return pd.DataFrame()

        query = (
            select(
                DimInstrument.ticker,
                DimInstrument.region,
                DimCalendar.date,
                FactFeaturesDaily.vrp_21d,
                FactFeaturesDaily.curve_slope,
                FactFeaturesDaily.skew_proxy,
                FactFeaturesDaily.z_vrp_21d,
                FactFeaturesDaily.z_curve_slope,
                FactFeaturesDaily.z_skew_proxy,
                FactFeaturesDaily.composite_dislocation_score,
                FactFeaturesDaily.regime_score,
            )
            .join(DimInstrument, DimInstrument.instrument_id == FactFeaturesDaily.instrument_id)
            .join(DimCalendar, DimCalendar.date_id == FactFeaturesDaily.date_id)
            .where(FactFeaturesDaily.date_id == max_date_id)
            .order_by(FactFeaturesDaily.composite_dislocation_score.desc())
        )

        result = session.execute(query).all()
        cols = [
            "ticker", "region", "date", "vrp_21d", "curve_slope", "skew_proxy",
            "z_vrp_21d", "z_curve_slope", "z_skew_proxy",
            "composite_dislocation_score", "regime_score",
        ]
        return pd.DataFrame(result, columns=cols)


class DislocationRepo:
    """Repository for fact_dislocation_events operations."""

    @staticmethod
    def insert_events(session: Session, events: list[dict]) -> int:
        """Insert dislocation events."""
        count = 0
        for evt in events:
            obj = FactDislocationEvents(**evt)
            session.add(obj)
            count += 1
        session.flush()
        logger.debug(f"Inserted {count} dislocation events")
        return count

    @staticmethod
    def get_events(
        session: Session,
        ticker: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Get dislocation events."""
        query = (
            select(
                DimCalendar.date,
                DimInstrument.ticker,
                DimInstrument.region,
                FactDislocationEvents.event_type,
                FactDislocationEvents.severity,
                FactDislocationEvents.rule_version,
                FactDislocationEvents.payload_json,
            )
            .join(DimCalendar, DimCalendar.date_id == FactDislocationEvents.date_id)
            .join(DimInstrument, DimInstrument.instrument_id == FactDislocationEvents.instrument_id)
        )
        if ticker:
            query = query.where(DimInstrument.ticker == ticker)
        if start_date:
            query = query.where(DimCalendar.date >= start_date)
        if end_date:
            query = query.where(DimCalendar.date <= end_date)
        query = query.order_by(DimCalendar.date.desc()).limit(limit)

        result = session.execute(query).all()
        cols = ["date", "ticker", "region", "event_type", "severity", "rule_version", "payload_json"]
        return pd.DataFrame(result, columns=cols)

    @staticmethod
    def get_latest_events(session: Session, limit: int = 50) -> pd.DataFrame:
        """Get the most recent events."""
        return DislocationRepo.get_events(session, limit=limit)


class BacktestRepo:
    """Repository for backtest tables."""

    @staticmethod
    def insert_trades(session: Session, trades_df: pd.DataFrame, date_id_map: dict[date, int]) -> int:
        """Insert backtest trades."""
        count = 0
        for _, row in trades_df.iterrows():
            d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            date_id = date_id_map.get(d)
            if date_id is None:
                continue

            obj = FactBacktestTrades(
                strategy_id=str(row["strategy_id"]),
                date_id=date_id,
                instrument_id=int(row["instrument_id"]),
                signal=float(row["signal"]) if pd.notna(row.get("signal")) else None,
                position=float(row["position"]) if pd.notna(row.get("position")) else None,
                fill_price=float(row["fill_price"]) if pd.notna(row.get("fill_price")) else None,
                pnl=float(row["pnl"]) if pd.notna(row.get("pnl")) else None,
                fees=float(row["fees"]) if pd.notna(row.get("fees")) else None,
                slippage=float(row["slippage"]) if pd.notna(row.get("slippage")) else None,
            )
            session.add(obj)
            count += 1

        session.flush()
        logger.debug(f"Inserted {count} backtest trades")
        return count

    @staticmethod
    def insert_perf(session: Session, perf_df: pd.DataFrame, date_id_map: dict[date, int]) -> int:
        """Insert backtest performance records."""
        count = 0
        for _, row in perf_df.iterrows():
            d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            date_id = date_id_map.get(d)
            if date_id is None:
                continue

            # Check for existing
            existing = session.execute(
                select(FactBacktestPerf).where(
                    FactBacktestPerf.strategy_id == str(row["strategy_id"]),
                    FactBacktestPerf.date_id == date_id,
                )
            ).scalar_one_or_none()

            data = {
                "pnl": float(row["pnl"]) if pd.notna(row.get("pnl")) else None,
                "cumulative_pnl": float(row["cumulative_pnl"]) if pd.notna(row.get("cumulative_pnl")) else None,
                "drawdown": float(row["drawdown"]) if pd.notna(row.get("drawdown")) else None,
                "exposure": float(row["exposure"]) if pd.notna(row.get("exposure")) else None,
                "turnover": float(row["turnover"]) if pd.notna(row.get("turnover")) else None,
                "regime_id": int(row["regime_id"]) if pd.notna(row.get("regime_id")) else None,
            }

            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
            else:
                obj = FactBacktestPerf(
                    strategy_id=str(row["strategy_id"]),
                    date_id=date_id,
                    **data,
                )
                session.add(obj)
            count += 1

        session.flush()
        logger.debug(f"Inserted {count} backtest perf rows")
        return count

    @staticmethod
    def get_perf(session: Session, strategy_id: str) -> pd.DataFrame:
        """Get backtest performance for a strategy."""
        query = (
            select(
                DimCalendar.date,
                FactBacktestPerf.pnl,
                FactBacktestPerf.cumulative_pnl,
                FactBacktestPerf.drawdown,
                FactBacktestPerf.exposure,
                FactBacktestPerf.turnover,
                FactBacktestPerf.regime_id,
            )
            .join(DimCalendar, DimCalendar.date_id == FactBacktestPerf.date_id)
            .where(FactBacktestPerf.strategy_id == strategy_id)
            .order_by(DimCalendar.date)
        )

        result = session.execute(query).all()
        cols = ["date", "pnl", "cumulative_pnl", "drawdown", "exposure", "turnover", "regime_id"]
        return pd.DataFrame(result, columns=cols)

    @staticmethod
    def get_trades(session: Session, strategy_id: str) -> pd.DataFrame:
        """Get backtest trades for a strategy."""
        query = (
            select(
                DimCalendar.date,
                DimInstrument.ticker,
                FactBacktestTrades.signal,
                FactBacktestTrades.position,
                FactBacktestTrades.fill_price,
                FactBacktestTrades.pnl,
                FactBacktestTrades.fees,
                FactBacktestTrades.slippage,
            )
            .join(DimCalendar, DimCalendar.date_id == FactBacktestTrades.date_id)
            .join(DimInstrument, DimInstrument.instrument_id == FactBacktestTrades.instrument_id)
            .where(FactBacktestTrades.strategy_id == strategy_id)
            .order_by(DimCalendar.date)
        )

        result = session.execute(query).all()
        cols = ["date", "ticker", "signal", "position", "fill_price", "pnl", "fees", "slippage"]
        return pd.DataFrame(result, columns=cols)

    @staticmethod
    def get_strategy_ids(session: Session) -> list[str]:
        """Get all unique strategy IDs."""
        result = session.execute(
            select(FactBacktestPerf.strategy_id).distinct()
        ).scalars().all()
        return list(result)


class RegimeRepo:
    """Repository for dim_regime operations."""

    @staticmethod
    def upsert_regimes(session: Session, regimes: list[dict]) -> dict[str, int]:
        """Upsert regime definitions. Returns name -> regime_id mapping."""
        result = {}
        for regime in regimes:
            existing = session.execute(
                select(DimRegime).where(DimRegime.name == regime["name"])
            ).scalar_one_or_none()

            if existing:
                existing.definition_json = regime.get("definition_json")
                result[existing.name] = existing.regime_id
            else:
                obj = DimRegime(**regime)
                session.add(obj)
                session.flush()
                result[obj.name] = obj.regime_id

        return result


class ManifestRepo:
    """Repository for run_manifest operations."""

    @staticmethod
    def insert_run(session: Session, manifest: dict) -> None:
        """Insert a run manifest record."""
        obj = RunManifest(**manifest)
        session.add(obj)
        session.flush()

    @staticmethod
    def update_status(session: Session, run_id: str, status: str) -> None:
        """Update the status of a run."""
        existing = session.execute(
            select(RunManifest).where(RunManifest.run_id == run_id)
        ).scalar_one_or_none()
        if existing:
            existing.status = status

    @staticmethod
    def get_latest_run(session: Session) -> RunManifest | None:
        """Get the most recent run manifest."""
        return session.execute(
            select(RunManifest).order_by(RunManifest.timestamp.desc()).limit(1)
        ).scalar_one_or_none()
