"""Database engine and session management."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from loguru import logger
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from vol_radar.db.models import Base


class Database:
    """Manages SQLite database connections and sessions."""

    def __init__(self, db_path: str | Path = ":memory:"):
        """Initialize database engine.

        Args:
            db_path: Path to SQLite database file, or ':memory:' for in-memory.
        """
        if str(db_path) == ":memory:":
            self._url = "sqlite:///:memory:"
        else:
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._url = f"sqlite:///{db_path}"

        self._engine = create_engine(
            self._url,
            echo=False,
            connect_args={"check_same_thread": False},
        )

        # Enable WAL mode and foreign keys for SQLite
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._session_factory = sessionmaker(bind=self._engine)
        logger.debug(f"Database initialized: {self._url}")

    def create_tables(self) -> None:
        """Create all tables defined in ORM models."""
        Base.metadata.create_all(self._engine)
        logger.info("Database tables created successfully")

    def drop_tables(self) -> None:
        """Drop all tables. Use with caution."""
        Base.metadata.drop_all(self._engine)
        logger.warning("All database tables dropped")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a transactional session scope.

        Usage:
            with db.get_session() as session:
                session.add(obj)
                # auto-commits on exit, rolls back on exception
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine for pd.read_sql() calls."""
        return self._engine
