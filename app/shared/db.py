from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import load_config


class Base(DeclarativeBase):
    pass


_cfg = load_config()
_engine = create_async_engine(_cfg.postgres_dsn_async.unicode_string(), echo=False)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession]:
    async with _session_factory() as session:
        yield session


async def init_db() -> None:
    from app.analysis.infrastructure import models as _analysis_models  # noqa: F401
    from app.image.infrastructure import models as _image_models  # noqa: F401
    from app.map.infrastructure import models as _map_models  # noqa: F401
    from app.user.infrastructure import models as _user_models  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
