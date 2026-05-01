"""
Database engine + session factory for TBot's central SQLite store.

Usage:
    from tbot.db.session import init_db, get_session

    init_db()                        # call once at startup — creates tables if missing

    with get_session() as session:
        session.add(some_orm_object)
        # commits automatically on exit; rolls back on exception
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from tbot.config import cfg
from tbot.db.models import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        db_path = Path(cfg.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode — allows concurrent reads during a write (safe for live + backtest)
        @event.listens_for(_engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return _engine


def _get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=_get_engine(), expire_on_commit=False)
    return _SessionFactory


def init_db() -> None:
    """Create all tables if they don't already exist. Safe to call repeatedly."""
    Base.metadata.create_all(_get_engine())


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy Session; commit on clean exit, rollback on exception."""
    session: Session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
