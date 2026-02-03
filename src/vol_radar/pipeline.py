"""End-to-end pipeline orchestrator."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from loguru import logger

from vol_radar.backtest.execution import ExecutionModel
from vol_radar.backtest.performance import PerformanceAttributor
from vol_radar.backtest.risk import RiskManager
from vol_radar.backtest.strategy import VRPStrategy
from vol_radar.config import Settings, load_config
from vol_radar.db.database import Database
from vol_radar.db.repos import (
    BacktestRepo,
    CalendarRepo,
    DislocationRepo,
    FeatureRepo,
    InstrumentRepo,
    ManifestRepo,
    RegimeRepo,
    PriceRepo,
    VolIndexRepo,
)
from vol_radar.features.builder import FeatureBuilder
from vol_radar.features.regime import RegimeClassifier
from vol_radar.ingest.base import FallbackClient
from vol_radar.ingest.cboe import CboeClient
from vol_radar.ingest.stooq import StooqClient
from vol_radar.ingest.yahoo import YahooClient
from vol_radar.models.ml import MLModelSuite
from vol_radar.models.scorer import DislocationScorer


class Pipeline:
    """Orchestrates the full data pipeline: ingest -> features -> model -> backtest."""

    def __init__(self, config: Settings):
        self.config = config
        self.db = Database(config.database.full_path)
        self.db.create_tables()

        # Data clients
        self._yahoo = YahooClient()
        self._stooq = StooqClient()
        self._cboe = CboeClient()
        self._client = FallbackClient([self._yahoo, self._stooq])

        # Feature builder
        self._feature_builder = FeatureBuilder(config)

        # Models
        self._scorer = DislocationScorer(config.dislocation)
        self._ml = MLModelSuite(config.ml)

        # Backtest
        self._strategy = VRPStrategy(config.backtest)
        self._execution = ExecutionModel(config.backtest)
        self._risk = RiskManager(config.backtest)
        self._performance = PerformanceAttributor()

    def run(self, mode: str = "full") -> str:
        """Run the pipeline.

        Args:
            mode: 'full', 'ingest', 'features', 'backtest'.

        Returns:
            Run ID.
        """
        run_id = f"{datetime.utcnow():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        logger.info(f"Pipeline starting: run_id={run_id}, mode={mode}")

        try:
            with self.db.get_session() as session:
                # Record run start
                ManifestRepo.insert_run(session, {
                    "run_id": run_id,
                    "timestamp": datetime.utcnow(),
                    "config_hash": self.config.config_hash,
                    "status": "running",
                })

            if mode in ("full", "ingest"):
                self._run_ingest()

            if mode in ("full", "features"):
                self._run_features()

            if mode == "full":
                self._run_ml()
                self._run_backtest()

            if mode == "backtest":
                self._run_backtest()

            with self.db.get_session() as session:
                ManifestRepo.update_status(session, run_id, "completed")

            logger.info(f"Pipeline completed: run_id={run_id}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            with self.db.get_session() as session:
                ManifestRepo.update_status(session, run_id, "failed")
            raise

        return run_id

    def _run_ingest(self) -> None:
        """Step 1: Ingest price and vol index data."""
        logger.info("=== INGEST PHASE ===")

        end_date = date.today()
        start_date = end_date - timedelta(days=365 * self.config.history.default_years)

        with self.db.get_session() as session:
            # Seed instruments
            instrument_dicts = [
                {
                    "ticker": inst.ticker,
                    "name": inst.name,
                    "asset_class": inst.asset_class,
                    "region": inst.region,
                    "country": inst.country,
                    "currency": inst.currency,
                    "venue": inst.venue or "",
                    "timezone": inst.timezone or "",
                }
                for inst in self.config.instruments
            ]
            InstrumentRepo.upsert_instruments(session, instrument_dicts)

            # Seed vol indices as instruments too
            vol_dicts = [
                {
                    "ticker": vi.ticker,
                    "name": vi.name,
                    "asset_class": "vol_index",
                    "region": vi.region,
                    "country": vi.region,
                    "currency": "USD",
                    "venue": "",
                    "timezone": "",
                }
                for vi in self.config.vol_indices
            ]
            InstrumentRepo.upsert_instruments(session, vol_dicts)

            # Seed regime definitions
            RegimeRepo.upsert_regimes(session, RegimeClassifier.define_regimes())

        # Fetch price data for each instrument
        total_price_rows = 0
        for inst in self.config.instruments:
            try:
                df = self._client.fetch_ohlcv(inst.ticker, start_date, end_date)
                if df.empty:
                    logger.warning(f"No price data for {inst.ticker}")
                    continue

                with self.db.get_session() as session:
                    inst_id = InstrumentRepo.get_instrument_id(session, inst.ticker)
                    if inst_id is None:
                        continue

                    dates = [d.date() if hasattr(d, "date") else d for d in df["date"]]
                    date_map = CalendarRepo.ensure_dates(session, dates)
                    count = PriceRepo.upsert_prices(session, inst_id, df, date_map)
                    total_price_rows += count

                logger.info(f"Ingested {len(df)} price rows for {inst.ticker}")

            except Exception as e:
                logger.error(f"Failed to ingest prices for {inst.ticker}: {e}")

        logger.info(f"Total price rows ingested: {total_price_rows}")

        # Fetch vol index data
        total_vol_rows = 0
        for vi in self.config.vol_indices:
            if vi.source == "upload":
                logger.info(f"Vol index {vi.ticker} requires manual upload, skipping")
                continue

            try:
                df = self._client.fetch_vol_index(vi.ticker, start_date, end_date)
                if df.empty:
                    logger.warning(f"No vol index data for {vi.ticker}")
                    continue

                with self.db.get_session() as session:
                    inst_id = InstrumentRepo.get_instrument_id(session, vi.ticker)
                    if inst_id is None:
                        continue

                    dates = [d.date() if hasattr(d, "date") else d for d in df["date"]]
                    date_map = CalendarRepo.ensure_dates(session, dates)
                    count = VolIndexRepo.upsert_vol_data(session, inst_id, df, date_map)
                    total_vol_rows += count

                logger.info(f"Ingested {len(df)} vol index rows for {vi.ticker}")

            except Exception as e:
                logger.error(f"Failed to ingest vol index {vi.ticker}: {e}")

        logger.info(f"Total vol index rows ingested: {total_vol_rows}")

        # Fetch VIX futures term structure
        try:
            vix_futures = self._cboe.fetch_vix_futures_term_structure()
            if not vix_futures.empty:
                logger.info(f"Fetched {len(vix_futures)} VIX futures rows from CBOE")
            else:
                logger.warning("No VIX futures data from CBOE")
        except Exception as e:
            logger.warning(f"CBOE VIX futures fetch failed: {e}")

    def _run_features(self) -> None:
        """Step 2-3: Compute features and detect dislocations."""
        logger.info("=== FEATURE PHASE ===")

        with self.db.get_session() as session:
            instruments = InstrumentRepo.get_all_instruments(session)
            instrument_map = InstrumentRepo.get_instrument_map(session)
            date_map = CalendarRepo.get_date_map(session)

            # Load all prices
            instrument_prices = {}
            for inst in instruments:
                if inst.asset_class == "vol_index":
                    continue
                prices_df = PriceRepo.get_prices(session, inst.instrument_id)
                if not prices_df.empty:
                    instrument_prices[inst.instrument_id] = prices_df

            # Load all vol index data
            instrument_vol = {}
            vol_index_mapping = {}
            for vi in self.config.vol_indices:
                vi_id = instrument_map.get(vi.ticker)
                if vi_id:
                    vol_df = VolIndexRepo.get_vol_data(session, vi_id)
                    if not vol_df.empty:
                        instrument_vol[vi_id] = vol_df
                        vol_index_mapping[vi.ticker] = vi_id

            # Build features
            features_all = self._feature_builder.build_all(
                session, instrument_prices, instrument_vol, vol_index_mapping
            )

            # Store features
            total_feature_rows = 0
            for inst_id, feat_df in features_all.items():
                if feat_df.empty:
                    continue
                count = FeatureRepo.upsert_features(session, inst_id, feat_df, date_map)
                total_feature_rows += count

            logger.info(f"Total feature rows stored: {total_feature_rows}")

            # Detect and store dislocation events
            events = self._feature_builder.detect_dislocations(
                features_all, session, date_map, instrument_map
            )
            if events:
                DislocationRepo.insert_events(session, events)
                logger.info(f"Stored {len(events)} dislocation events")

    def _run_ml(self) -> None:
        """Step 4: Train ML models with walk-forward."""
        logger.info("=== ML PHASE ===")

        with self.db.get_session() as session:
            instruments = InstrumentRepo.get_all_instruments(session)

            for inst in instruments:
                if inst.asset_class == "vol_index":
                    continue

                features_df = FeatureRepo.get_features(
                    session, instrument_ids=[inst.instrument_id]
                )
                if features_df.empty or len(features_df) < 300:
                    logger.debug(f"Not enough data for ML on {inst.ticker}")
                    continue

                try:
                    result = self._ml.walk_forward_train(features_df)
                    if result["metrics"]:
                        logger.info(
                            f"ML for {inst.ticker}: "
                            f"LR AUC={result['metrics'].get('lr_auc', 0):.3f}, "
                            f"GB AUC={result['metrics'].get('gb_auc', 0):.3f}"
                        )
                except Exception as e:
                    logger.warning(f"ML training failed for {inst.ticker}: {e}")

    def _run_backtest(self, strategy_id: str = "vrp_carry_v1") -> None:
        """Step 5: Run backtest simulation."""
        logger.info("=== BACKTEST PHASE ===")

        with self.db.get_session() as session:
            instruments = InstrumentRepo.get_all_instruments(session)
            instrument_map = InstrumentRepo.get_instrument_map(session)
            date_map = CalendarRepo.get_date_map(session)

            # Get regions for risk management
            instrument_regions = {
                i.instrument_id: i.region for i in instruments
                if i.asset_class != "vol_index"
            }

            # Load features for all equity instruments
            equity_inst_ids = [
                i.instrument_id for i in instruments
                if i.asset_class != "vol_index"
            ]

            features_df = FeatureRepo.get_features(session, instrument_ids=equity_inst_ids)
            if features_df.empty:
                logger.warning("No features available for backtest")
                return

            # Load prices for fill simulation
            prices = {}
            for inst in instruments:
                if inst.asset_class == "vol_index":
                    continue
                price_df = PriceRepo.get_prices(session, inst.instrument_id)
                if not price_df.empty:
                    prices[inst.instrument_id] = price_df

            # Generate signals per instrument per date
            all_signals = []
            for inst_id in equity_inst_ids:
                inst_features = features_df[features_df["instrument_id"] == inst_id].copy()
                if inst_features.empty:
                    continue

                signals = self._strategy.generate_signals(inst_features)
                signals["instrument_id"] = inst_id
                all_signals.append(signals)

            if not all_signals:
                logger.warning("No signals generated")
                return

            signals_df = pd.concat(all_signals, ignore_index=True)

            # Convert to positions
            positions = self._strategy.signals_to_positions(signals_df, instrument_regions)

            if positions.empty:
                logger.warning("No positions generated")
                return

            # Apply risk limits
            positions = self._risk.apply_limits(
                positions, instrument_regions=instrument_regions
            )

            # Simulate fills
            fills = self._execution.simulate_fills(
                positions, prices, self.config.backtest.initial_capital
            )

            if fills.empty:
                logger.warning("No fills generated")
                return

            # Compute daily PnL
            daily_pnl = self._compute_daily_pnl(fills, prices)

            if daily_pnl.empty:
                logger.warning("No PnL computed")
                return

            # Compute performance metrics
            equity = self._performance.compute_equity_curve(
                daily_pnl["pnl"], self.config.backtest.initial_capital
            )
            metrics = self._performance.compute_metrics(equity)
            dd_series = self._performance.compute_drawdown_series(equity)

            logger.info(
                f"Backtest results: Sharpe={metrics['sharpe']:.2f}, "
                f"MaxDD={metrics['max_drawdown']:.2%}, "
                f"Return={metrics['total_return']:.2%}"
            )

            # Store trades
            trades_for_db = fills.copy()
            trades_for_db["strategy_id"] = strategy_id
            trades_for_db["signal"] = 0.0
            trades_for_db["position"] = trades_for_db["position_value"]
            trades_for_db["pnl"] = 0.0  # Individual trade PnL
            trades_for_db["fees"] = trades_for_db["cost"]
            trades_for_db["slippage"] = 0.0
            BacktestRepo.insert_trades(session, trades_for_db, date_map)

            # Store daily performance
            perf_df = daily_pnl.copy()
            perf_df["strategy_id"] = strategy_id
            perf_df["cumulative_pnl"] = perf_df["pnl"].cumsum()
            perf_df["drawdown"] = dd_series.values[:len(perf_df)] if len(dd_series) >= len(perf_df) else 0
            perf_df["exposure"] = 0.0
            perf_df["turnover"] = 0.0
            perf_df["regime_id"] = None
            BacktestRepo.insert_perf(session, perf_df, date_map)

            logger.info(f"Backtest stored: {len(fills)} trades, {len(perf_df)} perf rows")

    def _compute_daily_pnl(
        self,
        fills: pd.DataFrame,
        prices: dict[int, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute daily portfolio PnL from fills and prices.

        Returns:
            DataFrame with date, pnl columns.
        """
        # Build position tracker
        all_dates = sorted(fills["date"].unique())
        position_values = {}  # date -> total position value
        prev_total = self.config.backtest.initial_capital

        daily_records = []
        current_holdings = {}  # inst_id -> (shares, last_price)

        for dt in all_dates:
            dt_fills = fills[fills["date"] == dt]

            # Update holdings from fills
            for _, fill in dt_fills.iterrows():
                inst_id = int(fill["instrument_id"])
                target_value = fill["position_value"]
                fill_price = fill["fill_price"]

                if fill_price > 0:
                    shares = target_value / fill_price
                    current_holdings[inst_id] = (shares, fill_price)

            # Compute current portfolio value using today's prices
            total_value = self.config.backtest.initial_capital
            for inst_id, (shares, _) in current_holdings.items():
                price_df = prices.get(inst_id)
                if price_df is not None and not price_df.empty:
                    today_price = price_df[price_df["date"] == dt]
                    if not today_price.empty:
                        total_value += shares * float(today_price["close"].values[0]) - shares * float(current_holdings[inst_id][1])

            # Subtract costs
            day_costs = dt_fills["cost"].sum() if not dt_fills.empty else 0

            pnl = total_value - prev_total - day_costs
            daily_records.append({"date": dt, "pnl": pnl})
            prev_total = total_value

        return pd.DataFrame(daily_records) if daily_records else pd.DataFrame(columns=["date", "pnl"])
