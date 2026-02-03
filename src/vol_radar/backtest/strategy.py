"""VRP-based trading strategy: signal generation and position sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from vol_radar.config import BacktestConfig


class VRPStrategy:
    """Generate trading signals from VRP z-scores, gated by regime and ML."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._strategy_cfg = config.strategy

    def generate_signals(
        self,
        features_df: pd.DataFrame,
        regime: pd.Series | None = None,
        ml_probs: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Generate raw trading signals from features.

        Signal logic:
        - Base signal = sign(vrp_21d) * min(|z_vrp_21d|, cap) / cap
        - Regime gating: scale down in risk_off regime
        - ML boost: scale by ML probability

        Args:
            features_df: Features with z_vrp_21d, vrp_21d, regime_score.
            regime: Regime label series (optional).
            ml_probs: ML probability series (optional).

        Returns:
            DataFrame with columns: date, instrument_id, signal.
        """
        cap = self._strategy_cfg.signal_cap

        # Base signal from VRP z-score
        vrp = features_df.get("vrp_21d", pd.Series(0, index=features_df.index)).fillna(0)
        z_vrp = features_df.get("z_vrp_21d", pd.Series(0, index=features_df.index)).fillna(0)

        signal = np.sign(vrp) * np.clip(z_vrp.abs(), 0, cap) / cap

        # Regime gating
        if regime is not None:
            regime_scale = regime.map({
                "risk_off": self.config.risk.regime_risk_off_scale,
                "carry_friendly": 1.0,
                "neutral": 0.8,
                "transition": 0.5,
            }).fillna(0.8)
            signal = signal * regime_scale
        elif "regime_score" in features_df.columns:
            # Continuous regime scaling
            rs = features_df["regime_score"].fillna(50) / 100.0
            # High regime score (risk-off) -> reduce signal
            regime_factor = 1.0 - 0.7 * rs  # 1.0 at calm, 0.3 at panic
            signal = signal * regime_factor

        # ML boost
        if ml_probs is not None:
            ml_boost = pd.Series(1.0, index=features_df.index)
            ml_boost[ml_probs > self._strategy_cfg.ml_threshold_high] = self._strategy_cfg.ml_boost_high
            ml_boost[ml_probs < self._strategy_cfg.ml_threshold_low] = self._strategy_cfg.ml_boost_low
            signal = signal * ml_boost

        result = pd.DataFrame({
            "date": features_df["date"],
            "signal": signal.values,
        })

        if "instrument_id" in features_df.columns:
            result["instrument_id"] = features_df["instrument_id"]

        return result

    def signals_to_positions(
        self,
        signals_df: pd.DataFrame,
        instrument_regions: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        """Convert signals to portfolio positions.

        Positive signal (IV > RV, carry) -> long equity ETF (short vol)
        Negative signal (RV > IV) -> short equity ETF (long vol)

        Positions are normalized to target gross exposure and region caps.

        Args:
            signals_df: DataFrame with date, instrument_id, signal.
            instrument_regions: Dict mapping instrument_id to region.

        Returns:
            DataFrame with date, instrument_id, position_weight.
        """
        if signals_df.empty:
            return pd.DataFrame(columns=["date", "instrument_id", "position_weight"])

        max_gross = self.config.risk.max_gross_exposure
        region_cap = self.config.risk.region_cap

        positions = []

        for dt, group in signals_df.groupby("date"):
            raw_weights = group["signal"].values
            inst_ids = group["instrument_id"].values

            # Normalize: scale so gross exposure = max_gross
            gross = np.abs(raw_weights).sum()
            if gross > 0:
                normalized = raw_weights * (max_gross / gross)
            else:
                normalized = raw_weights

            # Apply region caps
            if instrument_regions:
                for region in set(instrument_regions.values()):
                    region_mask = np.array([
                        instrument_regions.get(iid, "") == region for iid in inst_ids
                    ])
                    region_gross = np.abs(normalized[region_mask]).sum()
                    if region_gross > region_cap:
                        scale = region_cap / region_gross
                        normalized[region_mask] *= scale

            for iid, weight in zip(inst_ids, normalized):
                positions.append({
                    "date": dt,
                    "instrument_id": int(iid),
                    "position_weight": float(weight),
                })

        return pd.DataFrame(positions)
