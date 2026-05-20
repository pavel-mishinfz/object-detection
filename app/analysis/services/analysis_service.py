import uuid
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.entity.detection_result import DetectionResult, ObjectType
from app.analysis.exceptions import NoImagesError
from app.analysis.contracts import IImageReader
from app.analysis.interfaces.detection_engine import IDetectionEngine
from app.analysis.interfaces.repository import IDetectionResultRepository, IObjectTypeRepository


# --- Impure functions (commands) ---

async def run_analysis(
    area_id: UUID,
    image_reader: IImageReader,
    engine: IDetectionEngine,
    repo: IDetectionResultRepository,
    object_type_repo: IObjectTypeRepository,
    session: AsyncSession
) -> None:

    existing = await repo.find_by_area(area_id)
    if existing:
        return

    images = await image_reader.get_images_by_area(area_id)
    if not images:
        raise NoImagesError("Нет сохраненных снимков для указанного полигона")

    all_detections = await engine.detect_batch([img.path for img in images])
    now = datetime.now()
    for image, detections in zip(images, all_detections):
        for raw in detections:
            object_type = await object_type_repo.find_by_id(raw.object_type_id)
            if object_type is None:
                continue
            result = DetectionResult(
                id=uuid.uuid4(),
                area_id=area_id,
                image_id=image.id,
                geometry=raw.geo_polygon,
                score=raw.score,
                object_type=object_type,
                created_at=now,
            )
            await repo.save(result)
    await session.commit()

async def delete_results(
    area_id: UUID,
    repo: IDetectionResultRepository,
    session: AsyncSession
) -> None:
    await repo.delete_by_area(area_id)
    await session.commit()


# --- Impure functions (queries) ---

async def get_results(
    area_id: UUID,
    repo: IDetectionResultRepository,
) -> list[DetectionResult]:
    return await repo.find_by_area(area_id)
