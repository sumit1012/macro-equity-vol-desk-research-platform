"""Regime classification: risk-off, carry-friendly, neutral, transition."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from vol_radar.config import FeatureConfig


class RegimeClassifier:
    """Classifies market regime based on vol, drawdown, and correlation signals."""

    REGIMES = ["risk_off", "carry_friendly", "neutral", "transition"]

    def __init__(
        self,
        risk_off_threshold: float = 70.0,
        carry_threshold: float = 30.0,
    ):
        """Initialize classifier.

        Args:
            risk_off_threshold: Regime score above this = risk_off.
            carry_threshold: Regime score below this = carry_friendly.
        """
        self.risk_off_threshold = risk_off_threshold
        self.carry_threshold = carry_threshold

    def compute_regime_score(self, features_df: pd.DataFrame) -> pd.Series:
        """Compute a continuous regime score from 0 (calm) to 100 (panic).

        Components:
        - Drawdown depth (0-100 scale)
        - Vol spike: rv_5d / rv_63d ratio
        - Vol level: absolute rv_21d level

        Args:
            features_df: DataFrame with columns: rv_5d, rv_21d, rv_63d, drawdown

        Returns:
            Regime score series (0-100).
        """
        scores = pd.Series(50.0, index=features_df.index)

        # Drawdown component (0-40 points)
        if "drawdown" in features_df.columns:
            dd = features_df["drawdown"].fillna(0).abs()
            dd_score = np.clip(dd / 0.20 * 40, 0, 40)  # 20% DD = max score
            scores = scores + dd_score - 20  # center around 50

        # Vol spike component (0-30 points)
        if "rv_5d" in features_df.columns and "rv_63d" in features_df.columns:
            rv_63d_safe = features_df["rv_63d"].replace(0, np.nan)
            spike_ratio = (features_df["rv_5d"] / rv_63d_safe).fillna(1.0)
            spike_score = np.clip((spike_ratio - 1.0) / 1.0 * 30, -15, 30)
            scores = scores + spike_score

        # Absolute vol level component (0-30 points)
        if "rv_21d" in features_df.columns:
            rv_level = features_df["rv_21d"].fillna(0)
            vol_score = np.clip((rv_level - 0.15) / 0.25 * 30, -15, 30)
            scores = scores + vol_score

        scores = np.clip(scores, 0, 100)
        return scores

    def classify(self, features_df: pd.DataFrame) -> pd.Series:
        """Classify regime for each row.

        Args:
            features_df: Feature DataFrame.

        Returns:
            Series of regime labels.
        """
        score = self.compute_regime_score(features_df)

        conditions = [
            score >= self.risk_off_threshold,
            score <= self.carry_threshold,
            (score > self.carry_threshold) & (score < self.risk_off_threshold) & (score.diff().abs() > 10),
        ]
        choices = ["risk_off", "carry_friendly", "transition"]

        regime = pd.Series("neutral", index=features_df.index)
        for cond, label in zip(conditions, choices):
            regime = regime.where(~cond, label)

        return regime

    @staticmethod
    def define_regimes() -> list[dict]:
        """Return regime definitions for the dim_regime table."""
        return [
            {
                "name": "risk_off",
                "definition_json": json.dumps({
                    "description": "High vol, deep drawdown, correlation spike",
                    "score_range": "70-100",
                }),
            },
            {
                "name": "carry_friendly",
                "definition_json": json.dumps({
                    "description": "Low vol, no drawdown, normal correlations",
                    "score_range": "0-30",
                }),
            },
            {
                "name": "neutral",
                "definition_json": json.dumps({
                    "description": "Average vol environment",
                    "score_range": "30-70",
                }),
            },
            {
                "name": "transition",
                "definition_json": json.dumps({
                    "description": "Regime is changing rapidly",
                    "score_range": "30-70 with high rate of change",
                }),
            },
        ]
