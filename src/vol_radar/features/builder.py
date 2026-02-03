"""Feature builder: orchestrates all feature computation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from vol_radar.config import Settings
from vol_radar.db.database import Database
from vol_radar.db.repos import (
    CalendarRepo,
    DislocationRepo,
    FeatureRepo,
    InstrumentRepo,
    PriceRepo,
    VolIndexRepo,
)
from vol_radar.features.dividends import (
    detect_ex_div_dates,
    div_shock_flag,
    dividend_gap_magnitude,
    dividend_shock_window,
)
from vol_radar.features.flows import (
    dollar_volume,
    dollar_volume_shock,
    flow_regime_flag,
    impact_proxy,
)
from vol_radar.features.regime import RegimeClassifier
from vol_radar.features.skew import (
    downside_tail_intensity,
    skew_proxy,
    vol_of_vol,
)
from vol_radar.features.volatility import (
    drawdown,
    realized_vol,
    roll_down_proxy,
    term_structure_curvature,
    term_structure_slope,
    vrp,
    zscore,
)


class FeatureBuilder:
    """Orchestrates feature computation for all instruments."""

    def __init__(self, config: Settings):
        self.config = config
        self._regime_classifier = RegimeClassifier()

    def build_features_for_instrument(
        self,
        instrument_id: int,
        prices_df: pd.DataFrame,
        vol_df: pd.DataFrame | None = None,
        vol_front_df: pd.DataFrame | None = None,
        vol_back_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build all features for a single instrument.

        Args:
            instrument_id: DB instrument ID.
            prices_df: Price DataFrame (date, open, high, low, close, adj_close, volume, returns).
            vol_df: Vol index data (mid-term, e.g., VIX or VIX3M). Optional.
            vol_front_df: Front vol index (e.g., VIX9D). Optional.
            vol_back_df: Back vol index (e.g., VIX3M). Optional.

        Returns:
            DataFrame with all features aligned to dates.
        """
        if prices_df.empty:
            logger.warning(f"No prices for instrument {instrument_id}, skipping features")
            return pd.DataFrame()

        features = pd.DataFrame({"date": prices_df["date"]})
        returns = prices_df["returns"].astype(float)

        cfg = self.config.features

        # ── Realized Volatility ──
        for window in cfg.rv_windows:
            rv = realized_vol(returns, window)
            features[f"rv_{window}d"] = rv

        # ── Drawdown ──
        features["drawdown"] = drawdown(prices_df["adj_close"].astype(float))

        # ── VRP (requires vol index data) ──
        has_vol = vol_df is not None and not vol_df.empty
        if has_vol:
            # Merge vol data with price dates
            vol_merged = pd.merge(
                features[["date"]],
                vol_df[["date", "vol_level"]],
                on="date",
                how="left",
            )
            implied_vol = vol_merged["vol_level"].astype(float) / 100.0  # VIX is on % scale

            for horizon in cfg.vrp_horizons:
                rv_col = f"rv_{horizon}d"
                if rv_col in features.columns:
                    features[f"vrp_{horizon}d"] = vrp(implied_vol, features[rv_col].astype(float))
                else:
                    features[f"vrp_{horizon}d"] = np.nan
        else:
            for horizon in cfg.vrp_horizons:
                features[f"vrp_{horizon}d"] = np.nan

        # ── Term Structure ──
        has_front = vol_front_df is not None and not vol_front_df.empty
        has_back = vol_back_df is not None and not vol_back_df.empty

        if has_vol and has_front:
            front_merged = pd.merge(
                features[["date"]],
                vol_front_df[["date", "vol_level"]].rename(columns={"vol_level": "vol_front"}),
                on="date",
                how="left",
            )
            mid_merged = pd.merge(
                features[["date"]],
                vol_df[["date", "vol_level"]].rename(columns={"vol_level": "vol_mid"}),
                on="date",
                how="left",
            )
            front_vol = front_merged["vol_front"].astype(float)
            mid_vol = mid_merged["vol_mid"].astype(float)

            features["curve_slope"] = term_structure_slope(front_vol, mid_vol)
            features["roll_down_proxy"] = roll_down_proxy(front_vol, mid_vol)

            if has_back:
                back_merged = pd.merge(
                    features[["date"]],
                    vol_back_df[["date", "vol_level"]].rename(columns={"vol_level": "vol_back"}),
                    on="date",
                    how="left",
                )
                back_vol = back_merged["vol_back"].astype(float)
                features["curve_curvature"] = term_structure_curvature(
                    front_vol, mid_vol, back_vol
                )
            else:
                features["curve_curvature"] = np.nan
        else:
            features["curve_slope"] = np.nan
            features["curve_curvature"] = np.nan
            features["roll_down_proxy"] = np.nan

        # ── Skew Proxy ──
        di = downside_tail_intensity(
            returns, window=cfg.skew_window, threshold_std=cfg.skew_tail_threshold
        )
        vov = vol_of_vol(features["rv_21d"], window=cfg.vol_of_vol_window)
        features["vol_of_vol"] = vov
        features["skew_proxy"] = skew_proxy(
            di, vov,
            weights=tuple(cfg.skew_weights),
            zscore_lookback=cfg.zscore_lookback,
        )

        # ── Flows ──
        dv = dollar_volume(prices_df["close"].astype(float), prices_df["volume"].astype(float))
        dv_shock = dollar_volume_shock(
            dv, lookback=self.config.etf_flows.volume_shock_lookback
        )
        features["flow_dollar_volume_zscore"] = dv_shock
        features["flow_impact_proxy"] = impact_proxy(returns, dv)

        # ── Dividends ──
        ex_div = detect_ex_div_dates(
            prices_df["close"].astype(float),
            prices_df["adj_close"].astype(float),
        )
        div_window = dividend_shock_window(ex_div, window=self.config.dividends.gap_window_days)
        gap_mag = dividend_gap_magnitude(
            prices_df["open"].astype(float),
            prices_df["close"].astype(float).shift(1),
        )
        features["div_shock_flag"] = div_shock_flag(
            gap_mag, ex_div, threshold=self.config.dividends.shock_threshold
        )

        # ── Z-scores ──
        lookback = cfg.zscore_lookback
        features["z_vrp_21d"] = zscore(features["vrp_21d"], lookback=lookback)
        features["z_curve_slope"] = zscore(features["curve_slope"], lookback=lookback)
        features["z_skew_proxy"] = zscore(features["skew_proxy"], lookback=lookback)

        # ── Regime Score ──
        features["regime_score"] = self._regime_classifier.compute_regime_score(features)

        # ── Composite Dislocation Score ──
        weights = self.config.dislocation.scorer_weights
        features["composite_dislocation_score"] = (
            weights[0] * features["z_vrp_21d"].abs().fillna(0)
            + weights[1] * features["z_curve_slope"].abs().fillna(0)
            + weights[2] * features["z_skew_proxy"].abs().fillna(0)
        )

        return features

    def build_all(
        self,
        session,
        instrument_prices: dict[int, pd.DataFrame],
        instrument_vol: dict[int, pd.DataFrame] | None = None,
        vol_index_mapping: dict[str, int] | None = None,
    ) -> dict[int, pd.DataFrame]:
        """Build features for all instruments.

        Args:
            session: DB session (for instrument lookups).
            instrument_prices: Dict of instrument_id -> price DataFrame.
            instrument_vol: Dict of instrument_id -> vol index DataFrame.
            vol_index_mapping: Maps related_instrument ticker to vol index instrument_id.

        Returns:
            Dict of instrument_id -> feature DataFrame.
        """
        if instrument_vol is None:
            instrument_vol = {}
        if vol_index_mapping is None:
            vol_index_mapping = {}

        features_all = {}
        instruments = InstrumentRepo.get_all_instruments(session)
        instrument_map = {i.instrument_id: i for i in instruments}

        for inst_id, prices_df in instrument_prices.items():
            inst = instrument_map.get(inst_id)
            if inst is None:
                continue

            ticker = inst.ticker

            # Find associated vol indices
            vol_df = None
            vol_front_df = None
            vol_back_df = None

            # For US instruments, map to VIX family
            for vi_cfg in self.config.vol_indices:
                if vi_cfg.related_instrument == ticker:
                    vi_inst_id = vol_index_mapping.get(vi_cfg.ticker)
                    if vi_inst_id and vi_inst_id in instrument_vol:
                        vi_data = instrument_vol[vi_inst_id]
                        # Categorize by tenor
                        if "9D" in vi_cfg.ticker:
                            vol_front_df = vi_data
                        elif "3M" in vi_cfg.ticker or "VXV" in vi_cfg.ticker:
                            vol_back_df = vi_data
                        else:
                            vol_df = vi_data

            features = self.build_features_for_instrument(
                inst_id, prices_df,
                vol_df=vol_df,
                vol_front_df=vol_front_df,
                vol_back_df=vol_back_df,
            )

            if not features.empty:
                features_all[inst_id] = features
                logger.info(f"Built {len(features)} feature rows for {ticker}")

        return features_all

    def compute_cross_market_spreads(
        self,
        features: dict[int, pd.DataFrame],
        session,
    ) -> pd.DataFrame:
        """Compute cross-market VRP and curve spreads.

        Example: VRP_US - VRP_EU, curve_US - curve_EU.

        Returns:
            DataFrame with cross-market spread features.
        """
        instruments = InstrumentRepo.get_all_instruments(session)
        region_map = {i.instrument_id: i.region for i in instruments}
        ticker_map = {i.instrument_id: i.ticker for i in instruments}

        # Find representative instruments per region
        region_reps = {}
        preferred = {"US": "SPY", "Europe": "VGK", "Asia": "EWJ"}
        for inst_id, ticker in ticker_map.items():
            region = region_map.get(inst_id, "")
            if ticker in preferred.values():
                region_reps[region] = inst_id

        if len(region_reps) < 2:
            logger.debug("Not enough regions for cross-market spreads")
            return pd.DataFrame()

        spreads = []
        regions = list(region_reps.keys())
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                r1, r2 = regions[i], regions[j]
                id1, id2 = region_reps[r1], region_reps[r2]

                if id1 not in features or id2 not in features:
                    continue

                f1 = features[id1].set_index("date")
                f2 = features[id2].set_index("date")

                # Align on common dates
                common_dates = f1.index.intersection(f2.index)
                if len(common_dates) == 0:
                    continue

                spread_df = pd.DataFrame({"date": common_dates})
                spread_df[f"vrp_spread_{r1}_{r2}"] = (
                    f1.loc[common_dates, "vrp_21d"].values
                    - f2.loc[common_dates, "vrp_21d"].values
                )
                spread_df[f"curve_spread_{r1}_{r2}"] = (
                    f1.loc[common_dates, "curve_slope"].values
                    - f2.loc[common_dates, "curve_slope"].values
                )
                spreads.append(spread_df)

        if not spreads:
            return pd.DataFrame()

        result = spreads[0]
        for s in spreads[1:]:
            result = pd.merge(result, s, on="date", how="outer")

        return result.sort_values("date").reset_index(drop=True)

    def detect_dislocations(
        self,
        features: dict[int, pd.DataFrame],
        session,
        date_id_map: dict,
        instrument_map: dict[str, int],
    ) -> list[dict]:
        """Detect dislocation events based on threshold rules.

        Returns:
            List of event dicts ready for DislocationRepo.insert_events.
        """
        events = []
        z_threshold = self.config.dislocation.zscore_threshold
        inv_persistence = self.config.dislocation.inversion_persistence

        instruments = InstrumentRepo.get_all_instruments(session)
        id_to_ticker = {i.instrument_id: i.ticker for i in instruments}

        for inst_id, feat_df in features.items():
            if feat_df.empty:
                continue

            ticker = id_to_ticker.get(inst_id, "unknown")

            for idx, row in feat_df.iterrows():
                d = row["date"].date() if hasattr(row["date"], "date") else row["date"]
                date_id = date_id_map.get(d)
                if date_id is None:
                    continue

                triggered = []

                # VRP extreme
                z_vrp = row.get("z_vrp_21d", 0)
                if pd.notna(z_vrp) and abs(z_vrp) > z_threshold:
                    triggered.append({
                        "event_type": "vrp_extreme",
                        "severity": float(abs(z_vrp)),
                        "detail": f"VRP z-score: {z_vrp:.2f}",
                    })

                # Curve inversion
                curve = row.get("curve_slope", np.nan)
                if pd.notna(curve) and curve < 0:
                    # Check persistence
                    recent_slopes = feat_df.loc[
                        max(0, idx - inv_persistence + 1):idx, "curve_slope"
                    ]
                    if (recent_slopes < 0).all() and len(recent_slopes) >= inv_persistence:
                        triggered.append({
                            "event_type": "curve_inversion",
                            "severity": float(abs(curve)),
                            "detail": f"Curve inverted for {inv_persistence}+ days, slope: {curve:.2f}",
                        })

                # Skew extreme
                z_skew = row.get("z_skew_proxy", 0)
                if pd.notna(z_skew) and abs(z_skew) > z_threshold:
                    triggered.append({
                        "event_type": "skew_extreme",
                        "severity": float(abs(z_skew)),
                        "detail": f"Skew z-score: {z_skew:.2f}",
                    })

                for event in triggered:
                    events.append({
                        "date_id": date_id,
                        "instrument_id": inst_id,
                        "event_type": event["event_type"],
                        "severity": event["severity"],
                        "rule_version": "v1.0",
                        "payload_json": json.dumps({
                            "ticker": ticker,
                            "detail": event["detail"],
                            "composite_score": float(row.get("composite_dislocation_score", 0) or 0),
                            "regime_score": float(row.get("regime_score", 0) or 0),
                        }),
                    })

        logger.info(f"Detected {len(events)} dislocation events")
        return events
