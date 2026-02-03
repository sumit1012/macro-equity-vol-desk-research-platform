"""Execution model: cost simulation and fill logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from vol_radar.config import BacktestConfig


class ExecutionModel:
    """Simulates trade execution with realistic cost model."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        # Sort ADV buckets from highest to lowest
        self._adv_buckets = sorted(
            config.cost_model.adv_buckets,
            key=lambda b: b.get("min_adv", 0),
            reverse=True,
        )
        self._fixed_fee = config.cost_model.fixed_fee_per_share

    def get_spread_bps(self, adv: float) -> float:
        """Get bid-ask spread in bps based on average daily volume (dollar).

        Args:
            adv: Average daily dollar volume.

        Returns:
            Spread in basis points.
        """
        for bucket in self._adv_buckets:
            if adv >= bucket.get("min_adv", 0):
                return bucket.get("bps", 5)
        return 15  # Default for very illiquid

    def compute_costs(
        self,
        trade_values: pd.Series,
        adv_values: pd.Series,
    ) -> pd.Series:
        """Compute total transaction costs for a series of trades.

        Args:
            trade_values: Absolute dollar value of each trade.
            adv_values: Average daily dollar volume for each instrument.

        Returns:
            Total cost per trade in dollars.
        """
        spread_costs = pd.Series(0.0, index=trade_values.index)

        for idx in trade_values.index:
            adv = adv_values.loc[idx] if idx in adv_values.index else 0
            bps = self.get_spread_bps(adv)
            spread_costs.loc[idx] = trade_values.loc[idx] * bps / 10_000

        # Fixed fee (approximate as fraction of trade)
        fixed_costs = trade_values * self._fixed_fee / 100  # rough approximation

        return spread_costs + fixed_costs

    def simulate_fills(
        self,
        target_positions: pd.DataFrame,
        prices: dict[int, pd.DataFrame],
        capital: float,
    ) -> pd.DataFrame:
        """Simulate trade fills for a sequence of target positions.

        Args:
            target_positions: DataFrame with date, instrument_id, position_weight.
            prices: Dict of instrument_id -> price DataFrame with date, open, close, volume.
            capital: Total portfolio capital.

        Returns:
            DataFrame with trade details: date, instrument_id, fill_price, trade_value,
            cost, position_value, prev_position_value.
        """
        if target_positions.empty:
            return pd.DataFrame(columns=[
                "date", "instrument_id", "fill_price", "trade_value",
                "cost", "position_value", "prev_position_value",
            ])

        fill_method = self.config.fill_method
        trades = []

        # Track previous positions
        prev_positions = {}  # instrument_id -> position_value

        for dt, group in target_positions.sort_values("date").groupby("date"):
            for _, row in group.iterrows():
                inst_id = int(row["instrument_id"])
                target_weight = float(row["position_weight"])

                # Get price for fill
                price_df = prices.get(inst_id)
                if price_df is None or price_df.empty:
                    continue

                # Find the matching date
                if "date" in price_df.columns:
                    price_row = price_df[price_df["date"] == dt]
                    if price_row.empty:
                        continue

                    if fill_method == "next_open":
                        # Use next available open price
                        next_rows = price_df[price_df["date"] > dt].head(1)
                        if next_rows.empty:
                            fill_price = float(price_row["close"].values[0])
                        else:
                            fill_price = float(next_rows["open"].values[0])
                    else:
                        fill_price = float(price_row["close"].values[0])

                    # Compute ADV (trailing 63-day average dollar volume)
                    mask = price_df["date"] <= dt
                    recent = price_df[mask].tail(63)
                    adv = (recent["close"].astype(float) * recent["volume"].astype(float)).mean()
                else:
                    continue

                # Compute trade
                target_value = capital * target_weight
                prev_value = prev_positions.get(inst_id, 0.0)
                trade_value = target_value - prev_value

                # Compute cost
                bps = self.get_spread_bps(adv)
                cost = abs(trade_value) * bps / 10_000

                trades.append({
                    "date": dt,
                    "instrument_id": inst_id,
                    "fill_price": fill_price,
                    "trade_value": trade_value,
                    "cost": cost,
                    "position_value": target_value,
                    "prev_position_value": prev_value,
                })

                prev_positions[inst_id] = target_value

        return pd.DataFrame(trades)
