"""Risk management: exposure limits, vol targeting, drawdown controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from vol_radar.config import BacktestConfig


class RiskManager:
    """Applies risk limits to portfolio positions."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._risk = config.risk

    def apply_limits(
        self,
        positions: pd.DataFrame,
        portfolio_state: dict | None = None,
        instrument_regions: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        """Apply all risk limits to positions.

        Args:
            positions: DataFrame with date, instrument_id, position_weight.
            portfolio_state: Dict with cumulative_pnl, peak_equity, current_dd.
            instrument_regions: Dict mapping instrument_id to region.

        Returns:
            Risk-adjusted positions DataFrame.
        """
        if positions.empty:
            return positions

        result = positions.copy()

        # 1. Max gross exposure
        result = self._apply_gross_limit(result)

        # 2. Region caps
        if instrument_regions:
            result = self._apply_region_caps(result, instrument_regions)

        # 3. Drawdown controls
        if portfolio_state:
            result = self._apply_drawdown_controls(result, portfolio_state)

        return result

    def _apply_gross_limit(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Scale down positions if gross exposure exceeds limit."""
        max_gross = self._risk.max_gross_exposure

        for dt, group in positions.groupby("date"):
            gross = group["position_weight"].abs().sum()
            if gross > max_gross:
                scale = max_gross / gross
                positions.loc[group.index, "position_weight"] *= scale

        return positions

    def _apply_region_caps(
        self,
        positions: pd.DataFrame,
        instrument_regions: dict[int, str],
    ) -> pd.DataFrame:
        """Cap exposure by region."""
        cap = self._risk.region_cap

        for dt, group in positions.groupby("date"):
            for region in set(instrument_regions.values()):
                mask = group["instrument_id"].map(
                    lambda x: instrument_regions.get(x, "") == region
                )
                region_indices = group.index[mask]
                region_gross = positions.loc[region_indices, "position_weight"].abs().sum()

                if region_gross > cap:
                    scale = cap / region_gross
                    positions.loc[region_indices, "position_weight"] *= scale

        return positions

    def _apply_drawdown_controls(
        self,
        positions: pd.DataFrame,
        portfolio_state: dict,
    ) -> pd.DataFrame:
        """Apply soft and hard drawdown stops."""
        current_dd = portfolio_state.get("current_dd", 0)
        soft = self._risk.drawdown_soft
        hard = self._risk.drawdown_hard

        if current_dd <= hard:
            # Hard stop: go flat
            logger.warning(f"Hard drawdown stop triggered: DD={current_dd:.4f}")
            positions["position_weight"] = 0.0
        elif current_dd <= soft:
            # Soft stop: reduce by 50%
            logger.info(f"Soft drawdown stop: DD={current_dd:.4f}, reducing by 50%")
            positions["position_weight"] *= 0.5

        return positions

    def compute_portfolio_vol(
        self,
        positions: dict[int, float],
        returns_matrix: pd.DataFrame,
        window: int = 21,
    ) -> float:
        """Compute realized portfolio volatility.

        Args:
            positions: Dict of instrument_id -> weight.
            returns_matrix: DataFrame with instrument returns (columns = instrument_ids).
            window: Trailing window.

        Returns:
            Annualized portfolio volatility.
        """
        if not positions or returns_matrix.empty:
            return 0.0

        # Build weight vector
        instrument_ids = list(positions.keys())
        available = [iid for iid in instrument_ids if iid in returns_matrix.columns]

        if not available:
            return 0.0

        weights = np.array([positions[iid] for iid in available])
        ret_subset = returns_matrix[available].tail(window).values

        # Portfolio returns
        port_returns = ret_subset @ weights

        # Annualized vol
        if len(port_returns) > 1:
            return float(np.std(port_returns) * np.sqrt(252))
        return 0.0

    def apply_vol_targeting(
        self,
        positions: pd.DataFrame,
        portfolio_vol: float,
    ) -> pd.DataFrame:
        """Scale positions to target a specific portfolio volatility.

        Args:
            positions: Current positions.
            portfolio_vol: Current realized portfolio vol.

        Returns:
            Vol-targeted positions.
        """
        target = self._risk.vol_target

        if portfolio_vol <= 0 or np.isnan(portfolio_vol):
            return positions

        scale = target / portfolio_vol
        scale = np.clip(scale, 0.1, 3.0)  # Prevent extreme scaling

        result = positions.copy()
        result["position_weight"] *= scale
        return result
