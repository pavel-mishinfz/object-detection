import typing

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import models
from .. import schemas


async def get_object_tye(
        session: AsyncSession, object_type_id: int
    ) -> models.ObjectType | None:
    """
    Возвращает информацию о типе объекта
    """

    result = await session.execute(select(models.ObjectType) \
                                   .where(models.ObjectType.id == object_type_id) \
                                   .limit(1)
                                   )
    return result.scalars().one_or_none()


async def upsert_object_type(
        session: AsyncSession, object_type: schemas.object_type.ObjectTypeUpsert
    ) -> models.ObjectType | None:
    """
    Обновляет или добавляет тип объекта в базу
    """

    stm = insert(models.ObjectType).values(object_type.model_dump())
    stm = stm.on_conflict_do_update(
        constraint='object_type_pkey',
        set_={"name": object_type.name}
    )
    result = await session.execute(stm)

    await session.commit()
    if result:
        return await get_object_tye(session, object_type.id)
    return None
