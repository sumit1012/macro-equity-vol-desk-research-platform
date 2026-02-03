"""Performance measurement and attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class PerformanceAttributor:
    """Compute performance metrics, attribution, and robustness checks."""

    def compute_metrics(self, equity_curve: pd.Series) -> dict:
        """Compute standard performance metrics.

        Args:
            equity_curve: Cumulative PnL series.

        Returns:
            Dict of metrics.
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return self._empty_metrics()

        daily_returns = equity_curve.diff().dropna()
        if daily_returns.std() == 0:
            return self._empty_metrics()

        ann_factor = np.sqrt(252)

        # Sharpe ratio
        sharpe = (daily_returns.mean() / daily_returns.std()) * ann_factor

        # Sortino ratio (downside deviation)
        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() if len(downside) > 0 else daily_returns.std()
        sortino = (daily_returns.mean() / downside_std) * ann_factor if downside_std > 0 else 0

        # Max drawdown
        peak = equity_curve.expanding().max()
        dd = (equity_curve - peak) / peak.replace(0, np.nan)
        max_dd = dd.min() if not dd.empty else 0

        # CVaR (95%)
        var_95 = daily_returns.quantile(0.05)
        cvar_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else var_95

        # Hit rate
        hit_rate = (daily_returns > 0).mean()

        # Annualized return
        total_return = equity_curve.iloc[-1] / max(equity_curve.iloc[0], 1e-10) - 1
        n_years = len(equity_curve) / 252
        ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1 if n_years > 0 else 0

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_dd),
            "cvar_95": float(cvar_95),
            "hit_rate": float(hit_rate),
            "annualized_return": float(ann_return),
            "total_return": float(total_return),
            "calmar": float(calmar),
            "n_days": int(len(equity_curve)),
            "daily_vol": float(daily_returns.std() * ann_factor),
        }

    def _empty_metrics(self) -> dict:
        return {
            "sharpe": 0, "sortino": 0, "max_drawdown": 0, "cvar_95": 0,
            "hit_rate": 0, "annualized_return": 0, "total_return": 0,
            "calmar": 0, "n_days": 0, "daily_vol": 0,
        }

    def compute_equity_curve(
        self,
        daily_pnl: pd.Series,
        initial_capital: float = 1_000_000,
    ) -> pd.Series:
        """Compute cumulative equity curve from daily PnL.

        Args:
            daily_pnl: Daily profit/loss series.
            initial_capital: Starting capital.

        Returns:
            Equity curve series.
        """
        return initial_capital + daily_pnl.cumsum()

    def compute_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Compute running drawdown series.

        Args:
            equity_curve: Equity curve.

        Returns:
            Drawdown series (negative values).
        """
        peak = equity_curve.expanding().max()
        return (equity_curve - peak) / peak.replace(0, np.nan)

    def rolling_metrics(
        self, equity_curve: pd.Series, window: int = 63
    ) -> pd.DataFrame:
        """Compute rolling performance metrics.

        Args:
            equity_curve: Equity curve.
            window: Rolling window in days.

        Returns:
            DataFrame with rolling Sharpe, vol, drawdown.
        """
        daily_returns = equity_curve.diff()

        rolling_mean = daily_returns.rolling(window).mean()
        rolling_std = daily_returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

        rolling_vol = rolling_std * np.sqrt(252)

        dd = self.compute_drawdown_series(equity_curve)
        rolling_dd = dd.rolling(window).min()

        return pd.DataFrame({
            "rolling_sharpe": rolling_sharpe,
            "rolling_vol": rolling_vol,
            "rolling_max_dd": rolling_dd,
        })

    def attribute_by_region(
        self,
        trades_df: pd.DataFrame,
        instrument_regions: dict[int, str],
    ) -> pd.DataFrame:
        """Attribute PnL by region.

        Args:
            trades_df: Trades DataFrame with instrument_id, pnl.
            instrument_regions: instrument_id -> region mapping.

        Returns:
            DataFrame with region-level PnL summary.
        """
        if trades_df.empty:
            return pd.DataFrame(columns=["region", "total_pnl", "avg_daily_pnl", "n_trades"])

        trades_df = trades_df.copy()
        trades_df["region"] = trades_df["instrument_id"].map(instrument_regions).fillna("Unknown")

        summary = trades_df.groupby("region").agg(
            total_pnl=("pnl", "sum"),
            avg_daily_pnl=("pnl", "mean"),
            n_trades=("pnl", "count"),
        ).reset_index()

        return summary

    def attribute_by_regime(
        self,
        perf_df: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> pd.DataFrame:
        """Attribute PnL by market regime.

        Args:
            perf_df: Performance DataFrame with date, pnl.
            regime_labels: Series of regime labels aligned to dates.

        Returns:
            DataFrame with regime-level PnL summary.
        """
        if perf_df.empty:
            return pd.DataFrame(columns=["regime", "total_pnl", "avg_daily_pnl", "n_days", "sharpe"])

        combined = perf_df.copy()
        combined["regime"] = regime_labels.values[:len(combined)] if len(regime_labels) >= len(combined) else "unknown"

        results = []
        for regime, group in combined.groupby("regime"):
            daily_pnl = group["pnl"]
            sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

            results.append({
                "regime": regime,
                "total_pnl": daily_pnl.sum(),
                "avg_daily_pnl": daily_pnl.mean(),
                "n_days": len(group),
                "sharpe": sharpe,
            })

        return pd.DataFrame(results)

    def stress_test(
        self,
        perf_df: pd.DataFrame,
        regime_labels: pd.Series,
        stress_regimes: list[str] | None = None,
    ) -> dict:
        """Run stress tests on specific regime periods.

        Args:
            perf_df: Performance DataFrame.
            regime_labels: Regime labels.
            stress_regimes: Which regimes to stress test (default: risk_off).

        Returns:
            Dict with stress test metrics.
        """
        if stress_regimes is None:
            stress_regimes = ["risk_off"]

        results = {}
        for regime in stress_regimes:
            mask = regime_labels == regime
            if mask.sum() == 0:
                results[regime] = self._empty_metrics()
                continue

            stress_pnl = perf_df.loc[mask[:len(perf_df)], "pnl"] if "pnl" in perf_df.columns else pd.Series()
            if stress_pnl.empty:
                results[regime] = self._empty_metrics()
                continue

            equity = self.compute_equity_curve(stress_pnl)
            results[regime] = self.compute_metrics(equity)

        return results

    def sensitivity_analysis(
        self,
        run_fn,
        param_grid: dict[str, list],
    ) -> pd.DataFrame:
        """Run strategy across a grid of parameters.

        Args:
            run_fn: Function(params) -> metrics dict.
            param_grid: Dict of param_name -> list of values to test.

        Returns:
            DataFrame with one row per parameter combination.
        """
        import itertools

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            try:
                metrics = run_fn(params)
                metrics.update(params)
                results.append(metrics)
            except Exception as e:
                logger.warning(f"Sensitivity run failed for {params}: {e}")
                continue

        return pd.DataFrame(results)
