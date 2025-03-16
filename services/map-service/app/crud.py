import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .database import models
from .schemas import AreaIn

from geoalchemy2.shape import from_shape
from shapely.geometry import shape


async def create_area(
        db: AsyncSession, area_in: AreaIn
) -> models.Area:
    """
    Создает новый полигон в БД
    """

    # Преобразуем GeoJSON в Shapely-объект
    shapely_geometry = shape(area_in.geometry.model_dump())

    # Преобразуем Shapely-объект в формат, понятный базе данных
    db_geometry = from_shape(shapely_geometry, srid=4326)

    db_area = models.Area(
        user_id=area_in.user_id,
        geometry=db_geometry,
        name=area_in.name
    )

    db.add(db_area)
    await db.commit()
    await db.refresh(db_area)
    return db_area

async def get_area(
    db: AsyncSession, area_id: uuid.UUID
) -> models.Area | None:
    """
    Возвращает информацию о полигоне
    """
    pass