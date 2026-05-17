from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Config, load_config
from app.image.infrastructure.image_storage import FileImageStorage
from app.image.infrastructure.repository import ImageRepository
from app.image.infrastructure.sentinel_gateway import SentinelHubGateway
from app.shared.database import get_session


def get_image_repository(
    session: AsyncSession = Depends(get_session)
) -> ImageRepository:
    return ImageRepository(session)


def get_image_storage(
    cfg: Config = Depends(load_config)
) -> FileImageStorage:
    return FileImageStorage(
        temp_dir=cfg.sentinel_temp_dir,
        images_dir=cfg.sentinel_images_dir,
    )


def get_sentinel_gateway(
    cfg: Config = Depends(load_config)
) -> SentinelHubGateway:
    return SentinelHubGateway(
        client_id=cfg.sentinel_client_id,
        client_secret=cfg.sentinel_client_secret.get_secret_value(),
    )
