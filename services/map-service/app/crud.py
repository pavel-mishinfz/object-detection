import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .database import models
from .schemas import AreaIn, AreaUpdate

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
    result = await db.execute(select(models.Area) \
                              .filter(models.Area.id == area_id)
                              )
    return result.scalars().one_or_none()


async def get_list_areas(
    db: AsyncSession, user_id: uuid.UUID, skip: int = 0, limit: int = 100
) -> list[models.Area]:
    """
    Возвращает список полигонов текущего пользователя
    """
    result = await db.execute(select(models.Area) \
                              .filter(models.Area.user_id == user_id) \
                              .order_by(models.Area.created_at.desc()) \
                              .offset(skip) \
                              .limit(limit)
                              )
    return result.scalars().all()


async def update_area(
        db: AsyncSession, area_id: uuid.UUID, area_update: AreaUpdate
    ) -> models.Area | None:
    """
    Обновляет информацию о полигоне
    """

    # Преобразуем GeoJSON в WKBElement
    shapely_geometry = shape(area_update.geometry.model_dump())
    db_geometry = from_shape(shapely_geometry, srid=4326)
    area_update.geometry = db_geometry

    data_to_update = {
        'name': area_update.name,
        'geometry': area_update.geometry
    }

    result = await db.execute(update(models.Area) \
                              .where(models.Area.id == area_id) \
                              .values(data_to_update)
                              )
    await db.commit()

    if result:
        return await get_area(db, area_id)
    return None


async def delete_area(
        db: AsyncSession, area_id: uuid.UUID
) -> models.Area | None:
    """
    Удаляет информацию о полигоне
    """
    deleted_area = await get_area(db, area_id)
    await db.execute(delete(models.Area) \
                     .filter(models.Area.id == area_id)
                     )

    await db.commit()

    return deleted_area

