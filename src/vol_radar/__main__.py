"""CLI entry point for Vol Radar pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

from vol_radar.config import load_config


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config YAML file")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, config_path, verbose):
    """Vol Radar - Global Equity Volatility Dislocation Radar."""
    ctx.ensure_object(dict)
    setup_logging("DEBUG" if verbose else "INFO")

    try:
        ctx.obj["config"] = load_config(config_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--mode",
    default="full",
    type=click.Choice(["full", "ingest", "features", "backtest"]),
    help="Pipeline mode",
)
@click.pass_context
def run(ctx, mode):
    """Run the data pipeline.

    Modes:
        full     - Run complete pipeline (ingest + features + ML + backtest)
        ingest   - Only fetch and store price/vol data
        features - Only compute features (requires data)
        backtest - Only run backtest (requires features)
    """
    from vol_radar.pipeline import Pipeline

    config = ctx.obj["config"]
    pipeline = Pipeline(config)

    click.echo(f"Running pipeline in '{mode}' mode...")
    run_id = pipeline.run(mode=mode)
    click.echo(f"Pipeline complete. Run ID: {run_id}")


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.pass_context
def upload(ctx, csv_path):
    """Upload vol index data from a CSV file.

    Expected CSV format:
        date,ticker,vol_level,source,quality_flag
        2024-01-02,^VSTOXX,15.5,manual,ok
    """
    from vol_radar.db.database import Database
    from vol_radar.ingest.upload import CsvUploadHandler

    config = ctx.obj["config"]
    db = Database(config.database.full_path)
    db.create_tables()

    handler = CsvUploadHandler()

    try:
        df = handler.parse_vol_index_csv(csv_path)
        click.echo(f"Parsed {len(df)} rows from {csv_path}")

        tickers = df["ticker"].unique()
        click.echo(f"Tickers found: {', '.join(tickers)}")

        with db.get_session() as session:
            count = handler.store_uploaded_data(df, session)

        click.echo(f"Successfully stored {count} vol index rows.")

    except ValueError as e:
        click.echo(f"CSV validation error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Upload failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show pipeline status and database statistics."""
    from vol_radar.db.database import Database
    from vol_radar.db.repos import ManifestRepo, InstrumentRepo

    config = ctx.obj["config"]
    db_path = config.database.full_path

    if not db_path.exists():
        click.echo("Database not initialized. Run 'vol-radar run --mode ingest' first.")
        return

    db = Database(db_path)

    with db.get_session() as session:
        # Last run
        latest_run = ManifestRepo.get_latest_run(session)
        if latest_run:
            click.echo(f"Last run: {latest_run.run_id} ({latest_run.status}) at {latest_run.timestamp}")
        else:
            click.echo("No pipeline runs recorded.")

        # Instrument count
        instruments = InstrumentRepo.get_all_instruments(session)
        equity_count = sum(1 for i in instruments if i.asset_class != "vol_index")
        vol_count = sum(1 for i in instruments if i.asset_class == "vol_index")
        click.echo(f"Instruments: {equity_count} equity ETFs, {vol_count} vol indices")

        # Row counts
        from sqlalchemy import text
        for table in ["fact_price_daily", "fact_vol_index_daily", "fact_features_daily",
                       "fact_dislocation_events", "fact_backtest_trades", "fact_backtest_perf"]:
            try:
                count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                click.echo(f"  {table}: {count:,} rows")
            except Exception:
                click.echo(f"  {table}: not initialized")


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the database (create tables)."""
    from vol_radar.db.database import Database

    config = ctx.obj["config"]
    db = Database(config.database.full_path)
    db.create_tables()
    click.echo(f"Database initialized at {config.database.full_path}")


if __name__ == "__main__":
    cli()
