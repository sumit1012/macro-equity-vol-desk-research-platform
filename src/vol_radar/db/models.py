"""SQLAlchemy ORM models for the Vol Radar database."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ──────────────────────────────────────────────
# Dimension Tables
# ──────────────────────────────────────────────


class DimInstrument(Base):
    __tablename__ = "dim_instrument"

    instrument_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(50), nullable=False)
    region: Mapped[str] = mapped_column(String(50), nullable=False)
    country: Mapped[str] = mapped_column(String(10), nullable=False)
    currency: Mapped[str] = mapped_column(String(10), nullable=False)
    venue: Mapped[str] = mapped_column(String(50), nullable=True)
    timezone: Mapped[str] = mapped_column(String(50), nullable=True)

    def __repr__(self) -> str:
        return f"<DimInstrument(ticker={self.ticker}, region={self.region})>"


class DimCalendar(Base):
    __tablename__ = "dim_calendar"

    date_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False, index=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    is_month_end: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_quarter_end: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<DimCalendar(date={self.date})>"


class DimRegime(Base):
    __tablename__ = "dim_regime"

    regime_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    definition_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<DimRegime(name={self.name})>"


# ──────────────────────────────────────────────
# Fact Tables
# ──────────────────────────────────────────────


class FactPriceDaily(Base):
    __tablename__ = "fact_price_daily"

    instrument_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_instrument.instrument_id"), primary_key=True
    )
    date_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_calendar.date_id"), primary_key=True
    )
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    returns: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_fact_price_daily_inst_date", "instrument_id", "date_id"),
    )


class FactVolIndexDaily(Base):
    __tablename__ = "fact_vol_index_daily"

    instrument_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_instrument.instrument_id"), primary_key=True
    )
    date_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_calendar.date_id"), primary_key=True
    )
    vol_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str | None] = mapped_column(String(50), nullable=True)
    quality_flag: Mapped[str | None] = mapped_column(String(20), nullable=True)

    __table_args__ = (
        Index("ix_fact_vol_index_inst_date", "instrument_id", "date_id"),
    )


class FactFeaturesDaily(Base):
    __tablename__ = "fact_features_daily"

    instrument_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_instrument.instrument_id"), primary_key=True
    )
    date_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_calendar.date_id"), primary_key=True
    )
    # Realized volatility
    rv_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    rv_21d: Mapped[float | None] = mapped_column(Float, nullable=True)
    rv_63d: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Variance risk premium
    vrp_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    vrp_21d: Mapped[float | None] = mapped_column(Float, nullable=True)
    vrp_63d: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Term structure
    curve_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    curve_curvature: Mapped[float | None] = mapped_column(Float, nullable=True)
    roll_down_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Skew
    skew_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol_of_vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Drawdown
    drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Flows
    flow_dollar_volume_zscore: Mapped[float | None] = mapped_column(Float, nullable=True)
    flow_impact_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Dividends
    div_shock_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    # Regime
    regime_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Z-scores
    z_vrp_21d: Mapped[float | None] = mapped_column(Float, nullable=True)
    z_curve_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    z_skew_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Composite
    composite_dislocation_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_fact_features_inst_date", "instrument_id", "date_id"),
    )


class FactDislocationEvents(Base):
    __tablename__ = "fact_dislocation_events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_calendar.date_id"), nullable=False
    )
    instrument_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_instrument.instrument_id"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[float | None] = mapped_column(Float, nullable=True)
    rule_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_fact_dislocation_date_type", "date_id", "event_type"),
    )


class FactBacktestTrades(Base):
    __tablename__ = "fact_backtest_trades"

    trade_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(String(50), nullable=False)
    date_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_calendar.date_id"), nullable=False
    )
    instrument_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_instrument.instrument_id"), nullable=False
    )
    signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    position: Mapped[float | None] = mapped_column(Float, nullable=True)
    fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    fees: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_fact_backtest_trades_strat_date", "strategy_id", "date_id"),
    )


class FactBacktestPerf(Base):
    __tablename__ = "fact_backtest_perf"

    strategy_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    date_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_calendar.date_id"), primary_key=True
    )
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    cumulative_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    exposure: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("dim_regime.regime_id"), nullable=True
    )


# ──────────────────────────────────────────────
# Meta Tables
# ──────────────────────────────────────────────


class RunManifest(Base):
    __tablename__ = "run_manifest"

    run_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    git_hash: Mapped[str | None] = mapped_column(String(40), nullable=True)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    source_timestamps_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_checksums_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
