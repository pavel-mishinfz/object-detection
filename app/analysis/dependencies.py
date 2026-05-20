from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.contracts import IImageReader
from app.analysis.infrastructure.repository import ObjectTypeRepository, SegmentationResultRepository
from app.analysis.infrastructure.segmentation_engine import UNetSegmentationEngine
from app.analysis.interfaces.repository import IObjectTypeRepository, ISegmentationResultRepository
from app.analysis.interfaces.segmentation_engine import ISegmentationEngine
from app.config import Config, load_config
from app.image.dependencies import get_image_reader_adapter
from app.shared.database import get_session


def get_segmentation_engine(cfg: Config = Depends(load_config)) -> ISegmentationEngine:
    return UNetSegmentationEngine(cfg.model_dir)


def get_segmentation_result_repository(
    session: AsyncSession = Depends(get_session),
) -> ISegmentationResultRepository:
    return SegmentationResultRepository(session)


def get_object_type_repository(
    session: AsyncSession = Depends(get_session),
) -> IObjectTypeRepository:
    return ObjectTypeRepository(session)


def get_image_reader(
    adapter: IImageReader = Depends(get_image_reader_adapter),
) -> IImageReader:
    return adapter
