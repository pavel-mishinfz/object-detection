from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.application.interfaces import (
    IDetectionEngine,
    IDetectionResultRepository,
    IObjectTypeRepository,
)
from app.shared.contracts import IImageReader
from app.analysis.infrastructure.repository import (
    DetectionResultRepository,
    ObjectTypeRepository,
)
from app.image.dependencies import (  # re-export
    get_image_repository,
    get_image_storage,
    get_sentinel_gateway,
)
from app.image.infrastructure.adapters import ImageReaderAdapter
from app.map.dependencies import (  # re-export
    get_area_access_policy,
    get_area_reader,
)
from app.shared.database import get_session
from app.shared.dependencies import get_event_publisher  # re-export
from app.user.application.interfaces import IGroupRepository
from app.user.infrastructure.auth_backend import get_current_user_id  # re-export
from app.user.infrastructure.group_repository import GroupRepository


__all__ = [
    "get_current_user_id",
    "get_area_access_policy",
    "get_area_reader",
    "get_image_storage",
    "get_sentinel_gateway",
    "get_image_repository",
    "get_image_reader",
    "get_detection_result_repository",
    "get_object_type_repository",
    "get_group_repository",
    "get_detection_engine",
    "set_detection_engine",
    "get_event_publisher",
]


# --- Singleton: ML engine, инициализируется в lifespan ---

_detection_engine: IDetectionEngine | None = None


def set_detection_engine(engine: IDetectionEngine) -> None:
    global _detection_engine
    _detection_engine = engine


def get_detection_engine() -> IDetectionEngine:
    if _detection_engine is None:
        raise RuntimeError(
            "Detection engine is not initialized. Did lifespan startup run?"
        )
    return _detection_engine


# --- Image ---

def get_image_reader(
    session: AsyncSession = Depends(get_session),
) -> IImageReader:
    return ImageReaderAdapter(session)


# --- Analysis ---

def get_detection_result_repository(
    session: AsyncSession = Depends(get_session),
) -> IDetectionResultRepository:
    return DetectionResultRepository(session)


def get_object_type_repository(
    session: AsyncSession = Depends(get_session),
) -> IObjectTypeRepository:
    return ObjectTypeRepository(session)


# --- User ---

def get_group_repository(
    session: AsyncSession = Depends(get_session),
) -> IGroupRepository:
    return GroupRepository(session)
