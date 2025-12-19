import uuid

from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import models
from .. import schemas

from geoalchemy2 import WKBElement


async def create_analysis(
        db: AsyncSession, polygon_id: uuid.UUID, image_id: uuid.UUID, geometry: WKBElement, score: float, object_type_id: int
    ) -> models.DetectionResult | None:
    """
    Возвращает информацию о результате детектирования снимка
    """

    db_analysis = models.DetectionResult(
        image_id=image_id,
        polygon_id=polygon_id,
        geometry=geometry,
        score=score,
        object_type_id=object_type_id
    )

    db.add(db_analysis)
    await db.commit()
    await db.refresh(db_analysis)

    return db_analysis


async def get_analysis_results(
    db: AsyncSession, polygon_id: uuid.UUID, skip: int = 0, limit: int = 100
) -> list[models.DetectionResult]:
    """
    Возвращает список результатов анализа определенного полигона
    """
    result = await db.execute(select(models.DetectionResult) \
                              .filter(models.DetectionResult.polygon_id == polygon_id) \
                              .offset(skip) \
                              .limit(limit)
                              )
    return result.scalars().all()


async def delete_analysis_results(
    db: AsyncSession, polygon_id: uuid.UUID
) -> list[models.DetectionResult]:
    """
    Удаляет результаты анализа из БД
    """
    results_to_delete = await get_analysis_results(db, polygon_id)
    await db.execute(delete(models.DetectionResult) \
                            .filter(models.DetectionResult.polygon_id == polygon_id)
                            )
    await db.commit()
    return len(results_to_delete) > 0

