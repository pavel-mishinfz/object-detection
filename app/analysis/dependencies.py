from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Config, load_config
from app.shared.contracts import IImageReader
from app.shared.database import get_session
from app.image.infrastructure.adapters import ImageReaderAdapter
from app.analysis.infrastructure.detection_engine import YoloDetectionEngine
from app.analysis.infrastructure.repository import DetectionResultRepository, ObjectTypeRepository


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
    session: AsyncSession = Depends(get_session),
) -> IImageReader:
    return ImageReaderAdapter(session)
