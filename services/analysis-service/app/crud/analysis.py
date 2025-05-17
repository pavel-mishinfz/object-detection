import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import models

from geoalchemy2 import WKBElement


async def create_analysis(
        db: AsyncSession, polygon_id: uuid.UUID, image_id: uuid.UUID, geometry: WKBElement, object_type_id: int
    ) -> models.SegmentationResult | None:
    """
    Сохраняет информацию о результате сегментирования снимка
    """

    db_analysis = models.SegmentationResult(
        polygon_id=polygon_id,
        image_id=image_id,
        geometry=geometry,
        object_type_id=object_type_id
    )

    db.add(db_analysis)
    await db.commit()
    await db.refresh(db_analysis)

    return db_analysis


async def get_analysis_results(
    db: AsyncSession, polygon_id: uuid.UUID, skip: int = 0, limit: int = 100
) -> list[models.SegmentationResult]:
    """
    Возвращает список результатов анализа определенного полигона
    """
    result = await db.execute(select(models.SegmentationResult) \
                              .filter(models.SegmentationResult.polygon_id == polygon_id) \
                              .offset(skip) \
                              .limit(limit)
                              )
    return result.scalars().all()