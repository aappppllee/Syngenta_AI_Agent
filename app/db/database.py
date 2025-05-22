import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import DeclarativeMeta

from app.config import settings # Your application settings

logger = logging.getLogger(__name__)

# Create an asynchronous engine
# The URL should come from your settings (e.g., settings.DATABASE_URL)
try:
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG_MODE, # Log SQL queries if in debug mode
        pool_pre_ping=True # Test connections before handing them out
    )
    logger.info(f"Async SQLAlchemy engine created for URL: {settings.DATABASE_URL}")
except Exception as e:
    logger.critical(f"Failed to create async SQLAlchemy engine: {e}", exc_info=True)
    # Depending on your application's needs, you might want to exit or handle this error.
    # For now, we'll let it raise if engine creation fails critically.
    raise

# Create a sessionmaker for creating AsyncSession instances
# expire_on_commit=False is often recommended for FastAPI with async sessions
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False, # Consider autoflush=False for more control
    autocommit=False
)

# Base for declarative models
# All your ORM models (like SystemUser, Role, etc.) should inherit from this Base.
Base: DeclarativeMeta = declarative_base()

async def get_db_session() -> AsyncSession:
    """
    FastAPI dependency that provides an asynchronous database session.
    It ensures the session is properly closed after the request.
    """
    async_session = AsyncSessionLocal()
    try:
        yield async_session
        await async_session.commit() # Commit changes if no exceptions
    except Exception as e:
        await async_session.rollback() # Rollback on error
        logger.error(f"Database session error: {e}", exc_info=settings.DEBUG_MODE)
        raise # Re-raise the exception to be handled by FastAPI's error handlers
    finally:
        await async_session.close()

async def create_db_and_tables():
    """
    Creates all database tables defined by models inheriting from Base.
    This is typically called once on application startup.
    """
    async with engine.begin() as conn:
        try:
            # For pgvector, ensure the extension is created if not already.
            # This might be better handled at the database level or via migrations.
            # await conn.run_sync(lambda sync_conn: sync_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector")))
            
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables checked/created successfully based on Base.metadata.")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}", exc_info=settings.DEBUG_MODE)
            # Depending on the error, you might want to raise it or handle it.
            # If tables are crucial for startup, raising might be appropriate.

# Note: Your actual ORM models (SystemUser, Role, PolicyDocument, etc.)
# should be defined in app/models/db_models.py and should all inherit from this 'Base'.
# Example (in app/models/db_models.py):
# from app.db.database import Base
# class MyTable(Base):
#     __tablename__ = "mytable"
#     id = Column(Integer, primary_key=True)
#     ...
