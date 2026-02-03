"""Interpretable dislocation scoring system."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from vol_radar.config import DislocationConfig


class DislocationScorer:
    """Scores dislocations using weighted z-score composites with regime adjustment."""

    def __init__(self, config: DislocationConfig):
        self.config = config
        self.weights = config.scorer_weights  # [vrp, curve, skew]
        self.thresholds = config.severity_thresholds

    def score(self, features_df: pd.DataFrame) -> pd.Series:
        """Compute composite dislocation score (0-100).

        Score = weighted sum of absolute z-scores, scaled and regime-adjusted.

        Args:
            features_df: DataFrame with z_vrp_21d, z_curve_slope, z_skew_proxy, regime_score.

        Returns:
            Dislocation score (0-100).
        """
        z_vrp = features_df.get("z_vrp_21d", pd.Series(0, index=features_df.index)).fillna(0)
        z_curve = features_df.get("z_curve_slope", pd.Series(0, index=features_df.index)).fillna(0)
        z_skew = features_df.get("z_skew_proxy", pd.Series(0, index=features_df.index)).fillna(0)

        # Weighted absolute z-scores
        raw_score = (
            self.weights[0] * z_vrp.abs()
            + self.weights[1] * z_curve.abs()
            + self.weights[2] * z_skew.abs()
        )

        # Scale: z=3 maps to ~75, z=5 maps to ~100
        scaled = np.clip(raw_score / 4.0 * 100, 0, 100)

        # Regime adjustment: amplify in extreme regimes, dampen in neutral
        regime = features_df.get("regime_score", pd.Series(50, index=features_df.index)).fillna(50)
        regime_factor = 0.7 + 0.6 * (regime / 100.0)  # 0.7 to 1.3
        adjusted = np.clip(scaled * regime_factor, 0, 100)

        return adjusted

    def classify_severity(self, score: pd.Series) -> pd.Series:
        """Map dislocation score to severity labels.

        Args:
            score: Dislocation score (0-100).

        Returns:
            Series of severity labels: low, medium, high, extreme.
        """
        conditions = [
            score >= self.thresholds.get("extreme", 85),
            score >= self.thresholds.get("high", 65),
            score >= self.thresholds.get("medium", 40),
        ]
        choices = ["extreme", "high", "medium"]
        return pd.Series(
            np.select(conditions, choices, default="low"),
            index=score.index,
        )

    def generate_events(
        self,
        features_df: pd.DataFrame,
        date_id_map: dict,
        instrument_id: int,
        ticker: str,
        min_severity: str = "medium",
    ) -> list[dict]:
        """Generate dislocation event records for significant dislocations.

        Args:
            features_df: Feature DataFrame with dates.
            date_id_map: date -> date_id mapping.
            instrument_id: Instrument ID.
            ticker: Ticker symbol.
            min_severity: Minimum severity to generate events for.

        Returns:
            List of event dicts for DislocationRepo.insert_events.
        """
        score = self.score(features_df)
        severity = self.classify_severity(score)

        severity_order = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
        min_level = severity_order.get(min_severity, 1)

        events = []
        for idx, row in features_df.iterrows():
            sev = severity.loc[idx]
            if severity_order.get(sev, 0) < min_level:
                continue

            d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            date_id = date_id_map.get(d)
            if date_id is None:
                continue

            # Determine primary driver
            z_values = {
                "vrp_extreme": abs(row.get("z_vrp_21d", 0) or 0),
                "curve_dislocation": abs(row.get("z_curve_slope", 0) or 0),
                "skew_extreme": abs(row.get("z_skew_proxy", 0) or 0),
            }
            primary_type = max(z_values, key=z_values.get)

            events.append({
                "date_id": date_id,
                "instrument_id": instrument_id,
                "event_type": primary_type,
                "severity": float(score.loc[idx]),
                "rule_version": "v1.0",
                "payload_json": json.dumps({
                    "ticker": ticker,
                    "severity_label": sev,
                    "composite_score": float(score.loc[idx]),
                    "z_vrp_21d": float(row.get("z_vrp_21d", 0) or 0),
                    "z_curve_slope": float(row.get("z_curve_slope", 0) or 0),
                    "z_skew_proxy": float(row.get("z_skew_proxy", 0) or 0),
                    "regime_score": float(row.get("regime_score", 0) or 0),
                }),
            })

        logger.debug(f"Generated {len(events)} dislocation events for {ticker}")
        return events
