import uuid
from datetime import datetime
from uuid import UUID

from app.analysis.domain.segmentation_result import ObjectType, SegmentationResult
from app.analysis.domain.errors import NoImagesError
from app.analysis.application.interfaces import (
    IAreaAccessPolicy,
    ISegmentationEngine,
    ISegmentationResultRepository,
    IImageReader,
    IObjectTypeRepository,
)


# --- Команды (impure, -> None) ---

async def run_segmentation(
    area_id: UUID,
    user_id: UUID,
    model_name: str,
    area_access_policy: IAreaAccessPolicy,
    image_reader: IImageReader,
    engine: ISegmentationEngine,
    repo: ISegmentationResultRepository,
    object_type_repo: IObjectTypeRepository,
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)

    existing = await repo.find_by_area(area_id)
    if existing:
        return

    images = await image_reader.get_images_for_area(area_id)
    if not images:
        raise NoImagesError("Нет сохраненных снимков для указанного полигона")

    now = datetime.now()
    engine.load(model_name)
    for image in images:
        contours = engine.segment(image.path)
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


async def delete_segmentation_results(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: ISegmentationResultRepository,
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)
    await repo.delete_by_area(area_id)


# --- Запросы (impure, возвращают данные) ---

async def get_segmentation_results(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: ISegmentationResultRepository,
) -> list[SegmentationResult]:
    await area_access_policy.check_ownership(area_id, user_id)
    return await repo.find_by_area(area_id)
