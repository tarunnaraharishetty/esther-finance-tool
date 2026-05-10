"""Persistent storage: engine, session, schema bootstrap.

Backed by SQLAlchemy. SQLite by default; switch via DATABASE_URL. ORM models
live in :mod:`src.data.orm`; query/upsert helpers in :mod:`src.data.repositories`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

log = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: "Engine | None" = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> "Engine":
    global _engine
    if _engine is None:
        url = get_settings().database_url
        _engine = create_engine(url, future=True, echo=False)
        if _engine.dialect.name == "sqlite":
            # SQLite needs FK enforcement enabled per connection — ON DELETE
            # CASCADE relies on this and it's off by default.
            @event.listens_for(_engine, "connect")
            def _enable_sqlite_fks(dbapi_conn: object, _record: object) -> None:
                cursor = dbapi_conn.cursor()  # type: ignore[attr-defined]
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create tables. Replace with Alembic migrations in production."""
    # Importing orm registers tables on Base.metadata.
    from src.data import orm  # noqa: F401

    Base.metadata.create_all(bind=get_engine())


def reset_engine() -> None:
    """Drop the cached engine + session factory.

    Tests that swap DATABASE_URL must call this so a fresh engine is built
    against the new URL. Safe to call when no engine exists.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
