"""
Database connection and session management for the YouTube Video Summarizer.
"""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.config import config

# Create base class for SQLAlchemy models
Base = declarative_base()

# Create engine
# Use environment variable for database URL or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{config.DATA_DIR}/youtube_summarizer.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Type alias for session
DBSession = Session


def init_db() -> None:
    """Initialize the database by creating all tables."""
    from app.db.models import Video, Transcript, Summary, ChatHistory

    # Create all tables
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.

    This is a dependency that will be used in FastAPI route functions.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()