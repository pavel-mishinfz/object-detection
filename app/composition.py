from collections.abc import Callable

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.application.interfaces import (
    ISegmentationEngine,
    ISegmentationResultRepository,
    IImageReader,
    IObjectTypeRepository,
)
from app.analysis.infrastructure.repository import (
    SegmentationResultRepository,
    ObjectTypeRepository,
)
from app.config import Config, load_config
from app.image.application.interfaces import (
    IAreaAccessPolicy,
    IAreaReader,
    IImageRepository,
    IImageStorage,
    ISentinelGateway,
)
from app.image.infrastructure.image_reader import ImageReader
from app.image.infrastructure.image_storage import FileImageStorage
from app.image.infrastructure.repository import ImageRepository
from app.image.infrastructure.sentinel_gateway import SentinelHubGateway
from app.map.application.interfaces import IEventPublisher, IPolygonRepository
from app.map.infrastructure.area_access_policy import AreaAccessPolicy
from app.map.infrastructure.area_reader import AreaReader
from app.map.infrastructure.repository import PolygonRepository
from app.shared import event_bus
from app.shared.db import get_session
from app.user.application.interfaces import IGroupRepository
from app.user.infrastructure.auth_backend import get_current_user_id  # re-export
from app.user.infrastructure.group_repository import GroupRepository


__all__ = [
    "get_current_user_id",
    "get_polygon_repository",
    "get_area_access_policy",
    "get_area_reader",
    "get_image_storage",
    "get_sentinel_gateway",
    "get_image_repository",
    "get_image_reader",
    "get_segmentation_result_repository",
    "get_object_type_repository",
    "get_group_repository",
    "get_segmentation_engine",
    "set_segmentation_engine",
    "get_event_publisher",
]


# --- Singleton: ML engine, инициализируется в lifespan ---

_segmentation_engine: ISegmentationEngine | None = None


def set_segmentation_engine(engine: ISegmentationEngine) -> None:
    global _segmentation_engine
    _segmentation_engine = engine


def get_segmentation_engine() -> ISegmentationEngine:
    if _segmentation_engine is None:
        raise RuntimeError(
            "Segmentation engine is not initialized. Did lifespan startup run?"
        )
    return _segmentation_engine


# --- Event publisher ---

class SessionEventPublisher:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._post_commit: list[Callable] = []

    async def publish(self, event: object) -> None:
        callbacks = await event_bus.publish(event, self._session)
        self._post_commit.extend(callbacks)

    async def run_post_commit(self) -> None:
        for cb in self._post_commit:
            await cb()


def get_event_publisher(
    session: AsyncSession = Depends(get_session),
) -> IEventPublisher:
    return SessionEventPublisher(session)


# --- Map ---

def get_polygon_repository(
    session: AsyncSession = Depends(get_session),
) -> IPolygonRepository:
    return PolygonRepository(session)


def get_area_access_policy(
    session: AsyncSession = Depends(get_session),
) -> IAreaAccessPolicy:
    return AreaAccessPolicy(session)


def get_area_reader(
    session: AsyncSession = Depends(get_session),
) -> IAreaReader:
    return AreaReader(session)


# --- Image ---

def get_image_storage(cfg: Config = Depends(load_config)) -> IImageStorage:
    return FileImageStorage(
        temp_dir=cfg.sentinel_temp_dir,
        images_dir=cfg.sentinel_images_dir,
    )


def get_sentinel_gateway(cfg: Config = Depends(load_config)) -> ISentinelGateway:
    return SentinelHubGateway(
        client_id=cfg.sentinel_client_id,
        client_secret=cfg.sentinel_client_secret.get_secret_value(),
    )


def get_image_repository(
    session: AsyncSession = Depends(get_session),
) -> IImageRepository:
    return ImageRepository(session)


def get_image_reader(
    session: AsyncSession = Depends(get_session),
) -> IImageReader:
    return ImageReader(session)


# --- Analysis ---

def get_segmentation_result_repository(
    session: AsyncSession = Depends(get_session),
) -> ISegmentationResultRepository:
    return SegmentationResultRepository(session)


def get_object_type_repository(
    session: AsyncSession = Depends(get_session),
) -> IObjectTypeRepository:
    return ObjectTypeRepository(session)


# --- User ---

def get_group_repository(
    session: AsyncSession = Depends(get_session),
) -> IGroupRepository:
    return GroupRepository(session)
