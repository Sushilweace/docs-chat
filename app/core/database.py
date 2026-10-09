"""SQLAlchemy engine, session factory and declarative base."""

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()


def _prepare_sqlite_dir(url: str) -> None:
    """Create the parent directory of a file-based SQLite database."""
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)


_prepare_sqlite_dir(settings.database_url)

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. (Use Alembic migrations for real schema evolution.)"""
    from app.models import db as _models  # noqa: F401  (registers tables on Base)

    Base.metadata.create_all(bind=engine)
