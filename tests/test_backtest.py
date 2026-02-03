"""Tests for backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_radar.backtest.execution import ExecutionModel
from vol_radar.backtest.performance import PerformanceAttributor
from vol_radar.backtest.risk import RiskManager
from vol_radar.backtest.strategy import VRPStrategy
from vol_radar.config import BacktestConfig, CostModelConfig, RiskConfig, StrategyConfig


@pytest.fixture
def backtest_config():
    return BacktestConfig(
        initial_capital=1_000_000,
        rebalance="daily",
        fill_method="same_close",
        cost_model=CostModelConfig(
            adv_buckets=[
                {"min_adv": 1_000_000_000, "bps": 1},
                {"min_adv": 100_000_000, "bps": 3},
                {"min_adv": 10_000_000, "bps": 8},
                {"min_adv": 0, "bps": 15},
            ],
            fixed_fee_per_share=0.005,
        ),
        risk=RiskConfig(
            max_gross_exposure=1.0,
            region_cap=0.4,
            vol_target=0.10,
            drawdown_soft=-0.05,
            drawdown_hard=-0.10,
            regime_risk_off_scale=0.3,
        ),
        strategy=StrategyConfig(),
    )


class TestVRPStrategy:
    def test_positive_vrp_gives_positive_signal(self, backtest_config):
        """Positive VRP should generate positive signal (long equity = short vol)."""
        strategy = VRPStrategy(backtest_config)
        features = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=5),
            "vrp_21d": [0.01, 0.02, 0.015, 0.025, 0.01],
            "z_vrp_21d": [1.5, 2.5, 2.0, 3.0, 1.0],
            "regime_score": [30, 30, 30, 30, 30],
        })

        signals = strategy.generate_signals(features)
        assert (signals["signal"] > 0).all()

    def test_negative_vrp_gives_negative_signal(self, backtest_config):
        """Negative VRP should generate negative signal (short equity = long vol)."""
        strategy = VRPStrategy(backtest_config)
        features = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3),
            "vrp_21d": [-0.01, -0.02, -0.015],
            "z_vrp_21d": [-1.5, -2.5, -2.0],
            "regime_score": [30, 30, 30],
        })

        signals = strategy.generate_signals(features)
        assert (signals["signal"] < 0).all()

    def test_signal_capped(self, backtest_config):
        """Signal magnitude should not exceed 1.0 (before regime scaling)."""
        strategy = VRPStrategy(backtest_config)
        features = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3),
            "vrp_21d": [0.05, 0.05, 0.05],
            "z_vrp_21d": [10.0, 15.0, 20.0],  # Extreme z-scores
        })

        signals = strategy.generate_signals(features)
        # After regime scaling (factor ~0.79 for regime_score=50 default),
        # max should be < 1.0
        assert (signals["signal"].abs() <= 1.01).all()

    def test_positions_normalized(self, backtest_config):
        strategy = VRPStrategy(backtest_config)
        signals = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "instrument_id": [1, 2, 3],
            "signal": [0.5, -0.3, 0.2],
        })

        positions = strategy.signals_to_positions(signals)
        gross = positions["position_weight"].abs().sum()
        assert gross <= backtest_config.risk.max_gross_exposure + 0.001


class TestExecutionModel:
    def test_spread_bps_by_adv(self, backtest_config):
        """Cost should vary by ADV bucket."""
        model = ExecutionModel(backtest_config)

        assert model.get_spread_bps(2_000_000_000) == 1
        assert model.get_spread_bps(500_000_000) == 3
        assert model.get_spread_bps(50_000_000) == 8
        assert model.get_spread_bps(5_000_000) == 15

    def test_cost_increases_with_trade_size(self, backtest_config):
        """Larger trades should have higher absolute costs."""
        model = ExecutionModel(backtest_config)

        small_trades = pd.Series([10_000])
        large_trades = pd.Series([1_000_000])
        adv = pd.Series([500_000_000])

        small_cost = model.compute_costs(small_trades, adv)
        large_cost = model.compute_costs(large_trades, adv)

        assert large_cost.iloc[0] > small_cost.iloc[0]


class TestRiskManager:
    def test_gross_exposure_limit(self, backtest_config):
        """Positions should be scaled down if gross > max."""
        rm = RiskManager(backtest_config)
        positions = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")] * 3,
            "instrument_id": [1, 2, 3],
            "position_weight": [0.6, -0.5, 0.4],  # Gross = 1.5
        })

        result = rm.apply_limits(positions)
        gross = result["position_weight"].abs().sum()
        assert gross <= backtest_config.risk.max_gross_exposure + 0.001

    def test_hard_drawdown_stop(self, backtest_config):
        """Hard DD stop should go flat."""
        rm = RiskManager(backtest_config)
        positions = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "instrument_id": [1, 2],
            "position_weight": [0.5, -0.3],
        })

        result = rm.apply_limits(
            positions,
            portfolio_state={"current_dd": -0.12},  # Below hard stop (-0.10)
        )

        assert (result["position_weight"] == 0).all()

    def test_soft_drawdown_reduces(self, backtest_config):
        """Soft DD stop should reduce by 50%."""
        rm = RiskManager(backtest_config)
        positions = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "instrument_id": [1, 2],
            "position_weight": [0.4, -0.3],
        })

        result = rm.apply_limits(
            positions,
            portfolio_state={"current_dd": -0.07},  # Between soft and hard
        )

        assert result["position_weight"].iloc[0] == pytest.approx(0.2, abs=0.001)
        assert result["position_weight"].iloc[1] == pytest.approx(-0.15, abs=0.001)


class TestPerformanceAttributor:
    def test_sharpe_positive_for_profitable(self):
        """Positive daily PnL should give positive Sharpe."""
        pa = PerformanceAttributor()
        daily_pnl = pd.Series(np.random.normal(100, 50, 252))  # Positive avg
        equity = pa.compute_equity_curve(daily_pnl)
        metrics = pa.compute_metrics(equity)

        assert metrics["sharpe"] > 0
        assert metrics["hit_rate"] > 0.5

    def test_max_drawdown_correct(self):
        """Max DD should be correctly computed."""
        pa = PerformanceAttributor()
        # Equity: 100, 110, 105, 100, 120
        equity = pd.Series([100, 110, 105, 100, 120])
        metrics = pa.compute_metrics(equity)

        # Max DD = (100 - 110) / 110 = -0.0909
        assert metrics["max_drawdown"] == pytest.approx(-0.0909, abs=0.001)

    def test_rolling_metrics_shape(self):
        """Rolling metrics should have correct shape."""
        pa = PerformanceAttributor()
        np.random.seed(42)
        equity = pd.Series(1_000_000 + np.cumsum(np.random.normal(0, 1000, 200)))
        rolling = pa.rolling_metrics(equity, window=63)

        assert "rolling_sharpe" in rolling.columns
        assert "rolling_vol" in rolling.columns
        assert len(rolling) == len(equity)

    def test_deterministic_metrics(self):
        """Same input should give same metrics."""
        pa = PerformanceAttributor()
        np.random.seed(42)
        pnl = pd.Series(np.random.normal(50, 100, 252))
        equity1 = pa.compute_equity_curve(pnl)
        equity2 = pa.compute_equity_curve(pnl)

        metrics1 = pa.compute_metrics(equity1)
        metrics2 = pa.compute_metrics(equity2)

        assert metrics1["sharpe"] == metrics2["sharpe"]
        assert metrics1["max_drawdown"] == metrics2["max_drawdown"]
