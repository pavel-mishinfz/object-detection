from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Config, load_config
from app.analysis.contracts import IImageReader
from app.analysis.infrastructure.detection_engine import YoloDetectionEngine
from app.analysis.infrastructure.repository import DetectionResultRepository, ObjectTypeRepository
from app.image.dependencies import get_image_reader_adapter
from app.map.dependencies import get_area_access_policy_adapter
from app.shared.contracts import IAreaAccessPolicy
from app.shared.database import get_session


def get_detection_engine(
    cfg: Config = Depends(load_config)
) -> YoloDetectionEngine:
    return YoloDetectionEngine(cfg.model_path)


def get_detection_result_repository(
    session: AsyncSession = Depends(get_session),
) -> DetectionResultRepository:
    return DetectionResultRepository(session)


def get_object_type_repository(
    session: AsyncSession = Depends(get_session),
) -> ObjectTypeRepository:
    return ObjectTypeRepository(session)


def get_image_reader(
    adapter: IImageReader = Depends(get_image_reader_adapter),
) -> IImageReader:
    return adapter


def get_area_access_policy(
    adapter: IAreaAccessPolicy = Depends(get_area_access_policy_adapter),
) -> IAreaAccessPolicy:
    return adapter
