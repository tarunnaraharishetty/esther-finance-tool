"""Persistent storage for OHLCV bars + news articles.

Backed by SQLAlchemy. SQLite by default; switch via DATABASE_URL.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
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
    Base.metadata.create_all(bind=get_engine())
