"""Configuration management for Vol Radar pipeline."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve().parent
    # Walk up until we find pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent


@dataclass(frozen=True)
class DatabaseConfig:
    path: str = "data/vol_radar.db"

    @property
    def full_path(self) -> Path:
        return get_project_root() / self.path


@dataclass(frozen=True)
class InstrumentConfig:
    ticker: str
    name: str
    asset_class: str
    region: str
    country: str
    currency: str
    venue: str
    timezone: str


@dataclass(frozen=True)
class VolIndexConfig:
    ticker: str
    name: str
    region: str
    source: str  # yahoo, upload
    related_instrument: str


@dataclass(frozen=True)
class FeatureConfig:
    rv_windows: list[int] = field(default_factory=lambda: [5, 21, 63])
    vrp_horizons: list[int] = field(default_factory=lambda: [5, 21, 63])
    zscore_lookback: int = 252
    skew_tail_threshold: float = -2.0
    skew_window: int = 63
    vol_of_vol_window: int = 21
    skew_weights: list[float] = field(default_factory=lambda: [0.5, 0.5])


@dataclass(frozen=True)
class DislocationConfig:
    zscore_threshold: float = 2.5
    spread_percentile: int = 95
    inversion_persistence: int = 3
    scorer_weights: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    severity_thresholds: dict[str, int] = field(
        default_factory=lambda: {"medium": 40, "high": 65, "extreme": 85}
    )


@dataclass(frozen=True)
class CostModelConfig:
    adv_buckets: list[dict[str, Any]] = field(default_factory=list)
    fixed_fee_per_share: float = 0.005


@dataclass(frozen=True)
class RiskConfig:
    max_gross_exposure: float = 1.0
    region_cap: float = 0.4
    vol_target: float = 0.10
    drawdown_soft: float = -0.05
    drawdown_hard: float = -0.10
    regime_risk_off_scale: float = 0.3


@dataclass(frozen=True)
class StrategyConfig:
    signal_cap: float = 3.0
    ml_boost_high: float = 1.2
    ml_boost_low: float = 0.8
    ml_threshold_high: float = 0.6
    ml_threshold_low: float = 0.4


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000
    rebalance: str = "daily"
    fill_method: str = "next_open"
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)


@dataclass(frozen=True)
class MLConfig:
    models: list[str] = field(default_factory=lambda: ["logistic_regression", "gradient_boosting"])
    target_horizon: int = 21
    walk_forward_window: int = 252
    random_seed: int = 42
    gb_n_estimators: int = 100
    gb_max_depth: int = 3


@dataclass(frozen=True)
class FlowConfig:
    volume_shock_lookback: int = 63
    volume_shock_threshold: float = 2.0


@dataclass(frozen=True)
class DividendConfig:
    gap_window_days: int = 5
    shock_threshold: float = 0.02


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    structured: bool = True


@dataclass(frozen=True)
class HistoryConfig:
    default_years: int = 5


@dataclass(frozen=True)
class Settings:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    instruments: list[InstrumentConfig] = field(default_factory=list)
    vol_indices: list[VolIndexConfig] = field(default_factory=list)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    dislocation: DislocationConfig = field(default_factory=DislocationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    etf_flows: FlowConfig = field(default_factory=FlowConfig)
    dividends: DividendConfig = field(default_factory=DividendConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def config_hash(self) -> str:
        """Compute a hash of the config for reproducibility."""
        import json

        data = str(self)
        return hashlib.md5(data.encode()).hexdigest()


def _build_cost_model(raw: dict) -> CostModelConfig:
    return CostModelConfig(
        adv_buckets=raw.get("adv_buckets", []),
        fixed_fee_per_share=raw.get("fixed_fee_per_share", 0.005),
    )


def _build_risk(raw: dict) -> RiskConfig:
    return RiskConfig(**raw)


def _build_strategy(raw: dict) -> StrategyConfig:
    return StrategyConfig(**raw)


def _build_backtest(raw: dict) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=raw.get("initial_capital", 1_000_000),
        rebalance=raw.get("rebalance", "daily"),
        fill_method=raw.get("fill_method", "next_open"),
        cost_model=_build_cost_model(raw.get("cost_model", {})),
        risk=_build_risk(raw.get("risk", {})),
        strategy=_build_strategy(raw.get("strategy", {})),
    )


def load_config(path: str | None = None) -> Settings:
    """Load configuration from a YAML file.

    Args:
        path: Path to YAML config file. If None, uses config/default.yaml
              relative to the project root.

    Returns:
        Frozen Settings dataclass.
    """
    if path is None:
        config_path = get_project_root() / "config" / "default.yaml"
    else:
        config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    instruments = [
        InstrumentConfig(**inst) for inst in raw.get("instruments", [])
    ]

    vol_indices = [
        VolIndexConfig(**vi) for vi in raw.get("vol_indices", [])
    ]

    features_raw = raw.get("features", {})
    features = FeatureConfig(**features_raw)

    dislocation_raw = raw.get("dislocation", {})
    dislocation = DislocationConfig(**dislocation_raw)

    backtest = _build_backtest(raw.get("backtest", {}))

    ml_raw = raw.get("ml", {})
    ml = MLConfig(**ml_raw)

    flow_raw = raw.get("etf_flows", {})
    etf_flows = FlowConfig(**flow_raw)

    div_raw = raw.get("dividends", {})
    dividends = DividendConfig(**div_raw)

    logging_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(**logging_raw)

    db_raw = raw.get("database", {})
    database = DatabaseConfig(**db_raw)

    history_raw = raw.get("history", {})
    history = HistoryConfig(**history_raw)

    return Settings(
        database=database,
        history=history,
        instruments=instruments,
        vol_indices=vol_indices,
        features=features,
        dislocation=dislocation,
        backtest=backtest,
        ml=ml,
        etf_flows=etf_flows,
        dividends=dividends,
        logging=logging_cfg,
    )
