"""
Database configuration and session management.
"""

from sqlmodel import SQLModel, create_engine, Session
from app.config import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

def create_db_and_tables():
    """Create database tables."""
    SQLModel.metadata.create_all(engine)

def get_session():
    """Dependency to get database session."""
    with Session(engine) as session:
        yield session