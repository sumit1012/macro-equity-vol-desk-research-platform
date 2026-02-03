# Global Equity Volatility Dislocation Radar

A quantitative research platform that monitors variance risk premium (VRP), implied volatility term structure, and skew proxies across 13 equity ETFs spanning US, European, and Asian markets. The system detects statistical dislocations in real-time, classifies market regimes, trains walk-forward ML models, and backtests systematic vol-carry strategies with full risk management.

## Architecture

```
vol_radar/
  config.py          Frozen dataclass config from YAML
  db/
    models.py        9 SQLAlchemy 2.0 ORM tables (star schema)
    database.py      SQLite engine with WAL mode
    repos.py         Repository pattern (upsert, query)
  ingest/
    base.py          Abstract client + FallbackClient wrapper
    yahoo.py         yfinance OHLCV + vol index data
    stooq.py         Stooq CSV fallback
    cboe.py          CBOE VIX futures term structure
    upload.py        Manual CSV upload handler
  features/
    volatility.py    RV, VRP, term structure slope/curvature
    skew.py          Downside tail intensity, vol-of-vol, skew proxy
    regime.py        Regime classifier (risk-off / carry / neutral / transition)
    flows.py         Dollar volume shock, ETF flow impact proxy
    dividends.py     Ex-div detection, dividend shock windows
    builder.py       Feature orchestrator + cross-market spreads
  models/
    scorer.py        Z-score composite dislocation scoring
    ml.py            Walk-forward LogisticRegression + GradientBoosting
  backtest/
    strategy.py      VRP carry strategy with regime gating + ML boost
    execution.py     ADV-bucketed transaction cost model
    risk.py          Gross exposure, region caps, DD stops, vol targeting
    performance.py   Sharpe, Sortino, CVaR, attribution by region/regime

dashboard/            Streamlit multipage app (5 pages)
  app.py             Entry point with pipeline status
  pages/
    1_global_monitor       Dislocation leaderboard, treemap heatmap
    2_vol_term_structure   VIX curves, slope/curvature time series
    3_vrp_skew_lab         VRP bands, skew proxy, RV comparison
    4_dislocation_alerts   Event history, severity drill-down
    5_backtest_lab         Equity curve, rolling metrics, PnL attribution
```

## Key Features

- **Variance Risk Premium tracking** across multiple horizons (5D / 21D / 63D) with z-score normalization
- **Term structure analysis** with slope, curvature, and roll-down proxy from VIX futures
- **Skew proxy** combining downside tail intensity and vol-of-vol into a single z-scored metric
- **Regime classification** using drawdown depth, vol spikes, and vol level (score 0-100)
- **Walk-forward ML** with no look-ahead bias: binary target (next-21D RV > current RV), 9 features, expanding window
- **Dislocation detection** with 4 rule types: VRP extreme, curve inversion, curve dislocation, skew extreme
- **Backtest engine** with ADV-bucketed spread costs (1/3/8/15 bps tiers), drawdown stops, vol targeting, and region caps
- **Cross-market spread** analysis (US-EU, US-Asia VRP/curve differentials)

## Coverage

| Region | ETFs | Vol Index |
|--------|------|-----------|
| US | SPY, QQQ, IWM | ^VIX, ^VIX9D, ^VIX3M |
| Europe | VGK, FEZ, EWU, EWG, EWQ | ^VSTOXX |
| Asia | EWJ, FXI, EWH, EWY, INDA | ^JNIV, ^VHSI |

## Setup

```bash
# Clone and install
git clone https://github.com/yourusername/global-equity-volatility-dislocation-radar.git
cd global-equity-volatility-dislocation-radar
pip install -e .

# Initialize database and run full pipeline
python -m vol_radar init
python -m vol_radar run --mode full

# Launch dashboard
streamlit run dashboard/app.py
```

## CLI

```bash
python -m vol_radar run --mode full       # ingest + features + ML + backtest
python -m vol_radar run --mode ingest     # data ingestion only
python -m vol_radar run --mode features   # feature computation only
python -m vol_radar run --mode backtest   # backtest only
python -m vol_radar upload data/vol.csv   # upload vol index CSV
python -m vol_radar status                # show pipeline status
```

## Upload Format

For manual vol index data (e.g., VSTOXX, VHSI), provide a CSV with:

```
date,ticker,vol_level,source,quality_flag
2024-01-02,^VSTOXX,14.23,manual,verified
```

## Tests

```bash
pytest tests/ -v                # 69 tests
pytest tests/ -v --cov=vol_radar --cov-report=html
```

## Tech Stack

- **Data**: yfinance, Stooq CSV, CBOE free downloads
- **Database**: SQLAlchemy 2.0 ORM, SQLite (WAL mode)
- **ML**: scikit-learn (LogisticRegression, GradientBoostingClassifier)
- **Viz**: Plotly + Streamlit
- **CLI**: Click
- **Config**: YAML + frozen dataclasses
- **Logging**: Loguru

## License

MIT
