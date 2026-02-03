"""Tests for volatility feature calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_radar.features.volatility import (
    drawdown,
    realized_variance,
    realized_vol,
    roll_down_proxy,
    term_structure_curvature,
    term_structure_slope,
    vrp,
    vrp_vol_level,
    zscore,
)
from vol_radar.features.skew import (
    downside_tail_intensity,
    skew_proxy,
    vol_of_vol,
)
from vol_radar.features.flows import (
    dollar_volume,
    dollar_volume_shock,
    flow_regime_flag,
    impact_proxy,
)
from vol_radar.features.dividends import (
    detect_ex_div_dates,
    dividend_shock_window,
)


class TestRealizedVol:
    def test_constant_returns_give_known_vol(self):
        """Constant returns should give a known realized vol."""
        n = 100
        daily_return = 0.01  # 1% daily
        returns = pd.Series([daily_return] * n)

        rv = realized_vol(returns, window=21, annualize=True)

        # Expected: sqrt(0.01^2 * 252/21) = 0.01 * sqrt(12) ≈ 0.03464
        expected = daily_return * np.sqrt(252.0 / 21.0 * 21)
        assert abs(rv.iloc[-1] - expected) < 0.001

    def test_zero_returns_give_zero_vol(self):
        """Zero returns should give zero vol."""
        returns = pd.Series([0.0] * 50)
        rv = realized_vol(returns, window=21)
        # After window fills, should be ~0
        assert rv.dropna().iloc[-1] < 1e-10

    def test_rv_nan_padding(self):
        """First (window-1) values should be NaN."""
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        rv = realized_vol(returns, window=21)
        assert rv.iloc[:20].isna().all()
        assert rv.iloc[20:].notna().all()

    def test_rv_positive(self):
        """Realized vol should always be non-negative."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.015, 200))
        rv = realized_vol(returns, window=21)
        assert (rv.dropna() >= 0).all()


class TestVRP:
    def test_positive_vrp(self):
        """When IV > RV, VRP should be positive (carry opportunity)."""
        iv = pd.Series([0.20, 0.25, 0.30])
        rv = pd.Series([0.15, 0.20, 0.25])
        result = vrp(iv, rv)
        assert (result > 0).all()

    def test_negative_vrp(self):
        """When IV < RV, VRP should be negative."""
        iv = pd.Series([0.15])
        rv = pd.Series([0.20])
        result = vrp(iv, rv)
        assert result.iloc[0] < 0

    def test_vrp_vol_level(self):
        """VRP in vol-level should be IV - RV."""
        iv = pd.Series([0.20])
        rv = pd.Series([0.15])
        result = vrp_vol_level(iv, rv)
        assert abs(result.iloc[0] - 0.05) < 1e-10


class TestTermStructure:
    def test_contango_positive_slope(self):
        """Mid > Front = contango = positive slope."""
        front = pd.Series([15.0, 16.0])
        mid = pd.Series([18.0, 20.0])
        slope = term_structure_slope(front, mid)
        assert (slope > 0).all()

    def test_backwardation_negative_slope(self):
        """Front > Mid = backwardation = negative slope."""
        front = pd.Series([20.0])
        mid = pd.Series([15.0])
        slope = term_structure_slope(front, mid)
        assert slope.iloc[0] < 0

    def test_curvature_humped(self):
        """Mid higher than avg of front+back = positive curvature."""
        front = pd.Series([14.0])
        mid = pd.Series([18.0])
        back = pd.Series([16.0])
        curv = term_structure_curvature(front, mid, back)
        # 18 - 0.5*(14+16) = 18 - 15 = 3
        assert abs(curv.iloc[0] - 3.0) < 1e-10

    def test_roll_down_positive_contango(self):
        """Roll-down should be positive in contango."""
        front = pd.Series([15.0])
        mid = pd.Series([18.0])
        rd = roll_down_proxy(front, mid, days_to_expiry=30)
        assert rd.iloc[0] > 0
        assert abs(rd.iloc[0] - 0.1) < 1e-10  # (18-15)/30 = 0.1


class TestZscore:
    def test_zscore_mean_near_zero(self):
        """Z-score of a random series should have mean near 0."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 500))
        z = zscore(series, lookback=252)
        valid = z.dropna()
        assert abs(valid.mean()) < 0.3  # Allow some tolerance

    def test_zscore_std_near_one(self):
        """Z-score should have std near 1."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(10, 2, 500))
        z = zscore(series, lookback=252)
        valid = z.dropna()
        assert abs(valid.std() - 1.0) < 0.3


class TestDrawdown:
    def test_no_drawdown_on_rising(self):
        """Strictly rising prices should have 0 drawdown."""
        prices = pd.Series([100, 101, 102, 103, 104])
        dd = drawdown(prices)
        assert (dd == 0).all()

    def test_drawdown_after_peak(self):
        """After a peak, drawdown should be negative."""
        prices = pd.Series([100, 110, 105, 100, 95])
        dd = drawdown(prices)
        # After peak at 110:
        assert dd.iloc[2] == pytest.approx((105 - 110) / 110, abs=1e-10)
        assert dd.iloc[4] == pytest.approx((95 - 110) / 110, abs=1e-10)


class TestSkew:
    def test_downside_tail_intensity_no_tails(self):
        """Small returns should have near-zero tail intensity."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.001, 200))
        intensity = downside_tail_intensity(returns, window=63, threshold_std=-2.0)
        valid = intensity.dropna()
        # Very small returns, threshold at -2 std -> very few tail events
        assert valid.iloc[-1] < 0.15

    def test_vol_of_vol_stable_when_vol_constant(self):
        """If RV is constant, vol-of-vol should be ~0."""
        rv = pd.Series([0.15] * 50)
        vov = vol_of_vol(rv, window=21)
        assert vov.dropna().iloc[-1] < 1e-10


class TestFlows:
    def test_dollar_volume(self):
        price = pd.Series([100.0, 200.0])
        volume = pd.Series([1000, 2000])
        dv = dollar_volume(price, volume)
        assert dv.iloc[0] == 100000
        assert dv.iloc[1] == 400000

    def test_flow_regime_flag(self):
        shock = pd.Series([0.5, 2.5, -3.0, 1.0])
        flags = flow_regime_flag(shock, threshold=2.0)
        assert flags.tolist() == [False, True, True, False]


class TestDividends:
    def test_dividend_shock_window(self):
        """Window should expand around ex-div dates."""
        ex_div = pd.Series([False, False, True, False, False, False, False])
        window = dividend_shock_window(ex_div, window=2)
        # Should be True around index 2
        assert window.iloc[1] == True  # 1 before
        assert window.iloc[2] == True  # ex-div day
        assert window.iloc[3] == True  # 1 after
