from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import load_config
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


_cfg = load_config()
_engine = create_async_engine(_cfg.postgres_dsn_async.unicode_string(), echo=False)
session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession]:
    async with session_factory() as session:
        yield session


async def init_db() -> None:
    from app.analysis.models import segmentation_result as _analysis_models  # noqa: F401
    from app.image.models import image as _image_models  # noqa: F401
    from app.map.models import area as _map_area  # noqa: F401
    from app.user.models import group as _user_group_models  # noqa: F401
    from app.user.models import user as _user_models  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
