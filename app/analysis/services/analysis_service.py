import uuid
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.contracts import IImageReader
from app.analysis.entity.segmentation_result import SegmentationResult
from app.analysis.exceptions import NoImagesError
from app.analysis.interfaces.repository import IObjectTypeRepository, ISegmentationResultRepository
from app.analysis.interfaces.segmentation_engine import ISegmentationEngine


async def run_segmentation(
    area_id: UUID,
    model_name: str,
    image_reader: IImageReader,
    engine: ISegmentationEngine,
    repo: ISegmentationResultRepository,
    object_type_repo: IObjectTypeRepository,
    session: AsyncSession,
) -> None:
    existing = await repo.find_by_area(area_id)
    if existing:
        return

    images = await image_reader.get_images_by_area(area_id)
    if not images:
        raise NoImagesError("Нет сохраненных снимков для указанного полигона")
    
    await engine.load_model(model_name)
    all_contours = await engine.segment_batch([img.path for img in images])
    now = datetime.now()
    for image, contours in zip(images, all_contours):
        for raw in contours:
            object_type = await object_type_repo.find_by_id(raw.object_type_id)
            if object_type is None:
                continue
            result = SegmentationResult(
                id=uuid.uuid4(),
                area_id=area_id,
                image_id=image.id,
                geometry=raw.geo_polygon,
                object_type=object_type,
                created_at=now,
            )
            await repo.save(result)
    await session.commit()


async def delete_segmentation_results(
    area_id: UUID,
    repo: ISegmentationResultRepository,
    session: AsyncSession,
) -> None:
    await repo.delete_by_area(area_id)
    await session.commit()


async def get_segmentation_results(
    area_id: UUID,
    repo: ISegmentationResultRepository,
) -> list[SegmentationResult]:
    return await repo.find_by_area(area_id)
