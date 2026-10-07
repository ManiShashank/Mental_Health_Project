from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings

Base = declarative_base()

# Configure Async Engine
engine_kwargs = {}
if "sqlite" in settings.DATABASE_URL:
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    **engine_kwargs
)

# Async Session Factory
async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def init_db() -> None:
    """
    Initialize database schema creating all required tables and ensuring schema compatibility.
    """
    from app.db import models  # Ensure all models are registered with Base
    from sqlalchemy import text
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # SQLite lightweight column migration support
        if "sqlite" in settings.DATABASE_URL:
            # Check rooms columns
            res = await conn.execute(text("PRAGMA table_info(rooms)"))
            room_cols = [row[1] for row in res.fetchall()]
            if "status" not in room_cols:
                await conn.execute(text("ALTER TABLE rooms ADD COLUMN status VARCHAR(32) DEFAULT 'active'"))
            if "started_at" not in room_cols:
                await conn.execute(text("ALTER TABLE rooms ADD COLUMN started_at TIMESTAMP"))
            if "ended_at" not in room_cols:
                await conn.execute(text("ALTER TABLE rooms ADD COLUMN ended_at TIMESTAMP"))

            # Check messages columns
            res_m = await conn.execute(text("PRAGMA table_info(messages)"))
            msg_cols = [row[1] for row in res_m.fetchall()]
            if "audio_url" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN audio_url VARCHAR(256)"))
            if "audio_duration_seconds" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN audio_duration_seconds FLOAT"))
            if "audio_format" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN audio_format VARCHAR(32)"))
            if "audio_size_bytes" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN audio_size_bytes INTEGER"))

            # Milestone 5: Speech-to-Text columns
            if "transcript_text" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN transcript_text TEXT"))
            if "transcription_status" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN transcription_status VARCHAR(32)"))
            if "transcription_model" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN transcription_model VARCHAR(64)"))
            if "transcription_language" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN transcription_language VARCHAR(16)"))
            if "transcription_error" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN transcription_error TEXT"))
            if "transcribed_at" not in msg_cols:
                await conn.execute(text("ALTER TABLE messages ADD COLUMN transcribed_at TIMESTAMP"))


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an async database session.
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
