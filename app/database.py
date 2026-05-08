from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.analysis.infrastructure.models import Base as AnalysisBase
from app.config import load_config
from app.image.infrastructure.models import Base as ImageBase
from app.map.infrastructure.models import Base as MapBase
from app.user.infrastructure.models import Base as UserBase


cfg = load_config()
_engine = create_async_engine(cfg.postgres_dsn_async.unicode_string(), echo=False)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession]:
    async with _session_factory() as session:
        yield session


async def init_db() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(MapBase.metadata.create_all)
        await conn.run_sync(ImageBase.metadata.create_all)
        await conn.run_sync(UserBase.metadata.create_all)
        await conn.run_sync(AnalysisBase.metadata.create_all)
        await conn.execute(text(
            "ALTER TABLE image ADD CONSTRAINT IF NOT EXISTS fk_image_area "
            "FOREIGN KEY (area_id) REFERENCES area(id) ON DELETE CASCADE"
        ))
        await conn.execute(text(
            "ALTER TABLE detection_result ADD CONSTRAINT IF NOT EXISTS fk_dr_area "
            "FOREIGN KEY (area_id) REFERENCES area(id) ON DELETE CASCADE"
        ))
        await conn.execute(text(
            "ALTER TABLE detection_result ADD CONSTRAINT IF NOT EXISTS fk_dr_image "
            "FOREIGN KEY (image_id) REFERENCES image(id) ON DELETE CASCADE"
        ))
