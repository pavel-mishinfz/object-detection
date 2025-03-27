import uuid
import json
from typing import List, Tuple

from sqlalchemy import delete, select, update, text
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
