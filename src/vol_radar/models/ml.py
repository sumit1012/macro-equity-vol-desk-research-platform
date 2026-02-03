"""ML models: logistic regression and gradient boosting with walk-forward training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from vol_radar.config import MLConfig


class MLModelSuite:
    """Walk-forward ML models for volatility prediction."""

    FEATURE_COLUMNS = [
        "rv_5d", "rv_21d", "rv_63d", "vrp_21d",
        "curve_slope", "curve_curvature", "skew_proxy",
        "vol_of_vol", "flow_dollar_volume_zscore", "regime_score",
    ]

    def __init__(self, config: MLConfig):
        self.config = config
        self._models: dict[str, object] = {}

    def prepare_features(
        self, features_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target for ML training.

        Target: binary label — 1 if next 21D realized vol exceeds current implied vol proxy.
        Since we may not have implied vol for all instruments, we use VRP sign as proxy:
        target = 1 if rv_21d_forward > rv_21d_current (vol increasing).

        Args:
            features_df: Feature DataFrame.

        Returns:
            (X, y) tuple of feature matrix and target.
        """
        # Select available feature columns
        available = [c for c in self.FEATURE_COLUMNS if c in features_df.columns]
        if not available:
            logger.warning("No ML feature columns found")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Create target: forward 21D realized vol exceeds current
        horizon = self.config.target_horizon
        if "rv_21d" in features_df.columns:
            forward_rv = features_df["rv_21d"].shift(-horizon)
            current_rv = features_df["rv_21d"]
            target = (forward_rv > current_rv).astype(int)
        else:
            logger.warning("rv_21d not available for ML target")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Build feature matrix
        X = features_df[available].copy()

        # Drop columns that are entirely NaN (e.g., VRP when no vol index)
        X = X.dropna(axis=1, how="all")

        if X.empty:
            logger.warning("All ML feature columns are NaN")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Drop rows with NaN in remaining features or target
        valid = X.notna().all(axis=1) & target.notna()
        X = X[valid]
        y = target[valid]

        return X, y

    def walk_forward_train(
        self, features_df: pd.DataFrame
    ) -> dict:
        """Walk-forward training with no look-ahead.

        At each step t, train on [t-window:t], predict t+1.

        Args:
            features_df: Feature DataFrame (sorted by date).

        Returns:
            Dict with predictions, feature importance, and evaluation metrics.
        """
        X, y = self.prepare_features(features_df)

        if len(X) < self.config.walk_forward_window + 50:
            logger.warning(
                f"Not enough data for walk-forward: {len(X)} rows, "
                f"need {self.config.walk_forward_window + 50}"
            )
            return {"predictions": pd.DataFrame(), "feature_importance": pd.DataFrame(), "metrics": {}}

        window = self.config.walk_forward_window
        seed = self.config.random_seed

        predictions = []
        last_lr_model = None
        last_gb_model = None

        for t in range(window, len(X) - 1):
            # Training window
            X_train = X.iloc[t - window:t]
            y_train = y.iloc[t - window:t]

            # Test point (next day)
            X_test = X.iloc[t:t + 1]
            y_test = y.iloc[t:t + 1]

            # Skip if target has no variance
            if y_train.nunique() < 2:
                continue

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            row = {"index": X_test.index[0], "y_true": y_test.values[0]}

            # Logistic Regression
            if "logistic_regression" in self.config.models:
                try:
                    lr = LogisticRegression(
                        random_state=seed, max_iter=1000, C=1.0
                    )
                    lr.fit(X_train_scaled, y_train)
                    row["lr_prob"] = lr.predict_proba(X_test_scaled)[0, 1]
                    row["lr_pred"] = lr.predict(X_test_scaled)[0]
                    last_lr_model = lr
                except Exception as e:
                    logger.debug(f"LR failed at t={t}: {e}")
                    row["lr_prob"] = 0.5
                    row["lr_pred"] = 0

            # Gradient Boosting
            if "gradient_boosting" in self.config.models:
                try:
                    gb = GradientBoostingClassifier(
                        n_estimators=self.config.gb_n_estimators,
                        max_depth=self.config.gb_max_depth,
                        random_state=seed,
                    )
                    gb.fit(X_train_scaled, y_train)
                    row["gb_prob"] = gb.predict_proba(X_test_scaled)[0, 1]
                    row["gb_pred"] = gb.predict(X_test_scaled)[0]
                    last_gb_model = gb
                except Exception as e:
                    logger.debug(f"GB failed at t={t}: {e}")
                    row["gb_prob"] = 0.5
                    row["gb_pred"] = 0

            predictions.append(row)

        if not predictions:
            return {"predictions": pd.DataFrame(), "feature_importance": pd.DataFrame(), "metrics": {}}

        pred_df = pd.DataFrame(predictions).set_index("index")

        # Feature importance from last trained models
        importance_data = {}
        feature_names = list(X.columns)

        if last_lr_model is not None:
            importance_data["lr_coef"] = last_lr_model.coef_[0]
        if last_gb_model is not None:
            importance_data["gb_importance"] = last_gb_model.feature_importances_

        if importance_data:
            importance_df = pd.DataFrame(importance_data, index=feature_names)
        else:
            importance_df = pd.DataFrame()

        # Evaluation metrics
        metrics = self._evaluate(pred_df)

        self._models = {"logistic_regression": last_lr_model, "gradient_boosting": last_gb_model}

        logger.info(
            f"Walk-forward complete: {len(pred_df)} predictions, "
            f"LR AUC={metrics.get('lr_auc', 'N/A'):.3f}, "
            f"GB AUC={metrics.get('gb_auc', 'N/A'):.3f}"
        )

        return {
            "predictions": pred_df,
            "feature_importance": importance_df,
            "metrics": metrics,
        }

    def _evaluate(self, pred_df: pd.DataFrame) -> dict:
        """Compute evaluation metrics on walk-forward predictions."""
        metrics = {}
        y_true = pred_df["y_true"]

        for model_prefix, prob_col, pred_col in [
            ("lr", "lr_prob", "lr_pred"),
            ("gb", "gb_prob", "gb_pred"),
        ]:
            if prob_col not in pred_df.columns:
                continue

            probs = pred_df[prob_col]
            preds = pred_df[pred_col]

            try:
                metrics[f"{model_prefix}_auc"] = roc_auc_score(y_true, probs)
            except ValueError:
                metrics[f"{model_prefix}_auc"] = 0.5

            metrics[f"{model_prefix}_accuracy"] = accuracy_score(y_true, preds)
            metrics[f"{model_prefix}_brier"] = brier_score_loss(y_true, probs)
            metrics[f"{model_prefix}_hit_rate"] = (preds == y_true).mean()

        return metrics

    def get_ensemble_probability(self, pred_df: pd.DataFrame) -> pd.Series:
        """Get ensemble probability (average of LR and GB).

        Args:
            pred_df: Predictions DataFrame from walk_forward_train.

        Returns:
            Ensemble probability series.
        """
        probs = []
        if "lr_prob" in pred_df.columns:
            probs.append(pred_df["lr_prob"])
        if "gb_prob" in pred_df.columns:
            probs.append(pred_df["gb_prob"])

        if not probs:
            return pd.Series(0.5, index=pred_df.index)

        return pd.concat(probs, axis=1).mean(axis=1)
