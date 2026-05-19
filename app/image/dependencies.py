from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Config, load_config
from app.image.contracts import IAreaReader
from app.image.infrastructure.adapters import ImageReaderAdapter
from app.image.infrastructure.image_storage import LocalImageStorage
from app.image.infrastructure.repository import ImageRepository
from app.image.infrastructure.image_cache import RedisImageCache
from app.image.infrastructure.sentinel_gateway import SentinelHubGateway
from app.map.dependencies import get_area_reader_adapter
from app.shared.database import get_session


def get_image_repository(
    session: AsyncSession = Depends(get_session)
) -> ImageRepository:
    return ImageRepository(session)


def get_image_reader_adapter(
    session: AsyncSession = Depends(get_session),
) -> ImageReaderAdapter:
    return ImageReaderAdapter(session)


def create_image_storage(cfg: Config) -> LocalImageStorage:
    return LocalImageStorage(
        temp_dir=cfg.sentinel_temp_dir,
        images_dir=cfg.sentinel_images_dir,
    )


def get_image_storage(
    cfg: Config = Depends(load_config)
) -> LocalImageStorage:
    return create_image_storage(cfg)


def get_redis_cache(
    cfg: Config = Depends(load_config)
) -> RedisImageCache:
    return RedisImageCache(
        host=cfg.redis_host,
        port=cfg.redis_port,
        db=cfg.redis_db
    )


def get_sentinel_gateway(
    cfg: Config = Depends(load_config)
) -> SentinelHubGateway:
    return SentinelHubGateway(
        client_id=cfg.sentinel_client_id,
        client_secret=cfg.sentinel_client_secret.get_secret_value(),
    )


def get_area_reader(
    adapter: IAreaReader = Depends(get_area_reader_adapter),
) -> IAreaReader:
    return adapter
