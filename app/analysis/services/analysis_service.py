import uuid
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.entity.detection_result import DetectionResult, ObjectType
from app.analysis.exceptions import NoImagesError
from app.analysis.infrastructure.detection_engine import YoloDetectionEngine
from app.analysis.infrastructure.repository import DetectionResultRepository, ObjectTypeRepository
from app.shared.contracts import IAreaAccessPolicy, IImageReader


# --- Impure functions (commands) ---

async def run_analysis(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    image_reader: IImageReader,
    engine: YoloDetectionEngine,
    repo: DetectionResultRepository,
    object_type_repo: ObjectTypeRepository,
    session: AsyncSession
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)

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
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: DetectionResultRepository,
    session: AsyncSession
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)
    await repo.delete_by_area(area_id)
    await session.commit()


# --- Impure functions (queries) ---

async def get_results(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: DetectionResultRepository,
) -> list[DetectionResult]:
    await area_access_policy.check_ownership(area_id, user_id)
    return await repo.find_by_area(area_id)
