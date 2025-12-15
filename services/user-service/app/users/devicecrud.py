import typing
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from .database import models
from . import schemas


async def create_device(
        device: schemas.device.DeviceIn, session: AsyncSession
    ) -> models.DeviceFingerprint:
    """
    Создает запись fingerprint бразуера пользователя в базе
    """
    
    db_device = models.DeviceFingerprint(**device.model_dump())

    session.add(db_device)
    await session.commit()
    await session.refresh(db_device)
    return db_device

async def get_devices(
        session: AsyncSession, user_id: uuid.UUID, skip: int = 0, limit: int = 100
    ) -> typing.List[models.DeviceFingerprint]:
    """
    Возвращает список устройств пользователя
    """

    result = await session.execute(select(models.DeviceFingerprint) \
                                   .where(models.DeviceFingerprint.user_id == user_id)
                                   .offset(skip) \
                                   .limit(limit)
                                   )
    return result.scalars().all()