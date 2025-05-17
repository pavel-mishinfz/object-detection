import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import models
from .schemas import ImageIn


async def create_image(
        db: AsyncSession, image_in: ImageIn
) -> models.Image:
    """
    Создает запись о снимке в БД
    """
    db_image = models.Image(**image_in.model_dump())

    db.add(db_image)
    await db.commit()
    await db.refresh(db_image)

    return db_image

async def get_images(
    db: AsyncSession, area_id: uuid.UUID, skip: int = 0, limit: int = 100
) -> list[models.Image]:
    """
    Возвращает список изображений определенного полигона
    """
    result = await db.execute(select(models.Image) \
                              .filter(models.Image.area_id == area_id) \
                              .offset(skip) \
                              .limit(limit)
                              )
    return result.scalars().all()
