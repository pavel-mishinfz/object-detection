import uuid
from datetime import datetime
from uuid import UUID

from app.analysis.domain.detection_result import DetectionResult, ObjectType
from app.analysis.domain.errors import NoImagesError
from app.analysis.application.interfaces import (
    IAreaAccessPolicy,
    IDetectionEngine,
    IDetectionResultRepository,
    IImageReader,
    IObjectTypeRepository,
)


# --- Чистые функции (нет IO) ---

def build_detection_result(
    result_id: UUID,
    area_id: UUID,
    image_id: UUID,
    geo_polygon: tuple[tuple[float, float], ...],
    score: float,
    object_type: ObjectType,
    created_at: datetime,
) -> DetectionResult:
    return DetectionResult(
        id=result_id,
        area_id=area_id,
        image_id=image_id,
        geometry=geo_polygon,
        score=score,
        object_type=object_type,
        created_at=created_at,
    )


# --- Команды (impure, -> None) ---

async def run_analysis(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    image_reader: IImageReader,
    engine: IDetectionEngine,
    repo: IDetectionResultRepository,
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
    for image in images:
        detections = engine.detect(image.path)
        for raw in detections:
            object_type = await object_type_repo.find_by_id(raw.object_type_id)
            if object_type is None:
                continue
            result = build_detection_result(
                result_id=uuid.uuid4(),
                area_id=area_id,
                image_id=image.id,
                geo_polygon=raw.geo_polygon,
                score=raw.score,
                object_type=object_type,
                created_at=now,
            )
            await repo.save(result)


async def delete_results(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: IDetectionResultRepository,
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)
    await repo.delete_by_area(area_id)


# --- Запросы (impure, возвращают данные) ---

async def get_results(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: IDetectionResultRepository,
) -> list[DetectionResult]:
    await area_access_policy.check_ownership(area_id, user_id)
    return await repo.find_by_area(area_id)
