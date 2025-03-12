import typing

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from .database import models
from . import schemas


async def create_group(
        group: schemas.group.GroupCreate, session: AsyncSession
    ) -> models.Group:
    """
    Создает новую группу пользователей в базе
    """

    db_group = models.Group(
        name=group.name
    )

    session.add(db_group)
    await session.commit()
    await session.refresh(db_group)
    return db_group


async def get_groups(
        session: AsyncSession, skip: int = 0, limit: int = 100
    ) -> typing.List[models.Group]:
    """
    Возвращает информацию о группах пользователей
    """

    result = await session.execute(select(models.Group) \
                                   .offset(skip) \
                                   .limit(limit)
                                   )
    return result.scalars().all()


async def get_group(
        session: AsyncSession, group_id: int
    ) -> models.Group | None:
    """
    Возвращает информацию о группе пользователей
    """

    result = await session.execute(select(models.Group) \
                                   .where(models.Group.id == group_id) \
                                   .limit(1)
                                   )
    return result.scalars().one_or_none()


async def update_group(
        session: AsyncSession, group_id: int, group: schemas.group.GroupUpdate
    ) -> models.Group | None:
    """
    Обновляет группу пользователей в базе
    """

    result = await session.execute(update(models.Group) \
                                   .where(models.Group.id == group_id) \
                                   .values(group.model_dump())
                                   )
    await session.commit()
    if result:
        return await get_group(session, group_id)
    return None


async def upsert_group(
        session: AsyncSession, group: schemas.group.GroupUpsert
    ) -> models.Group | None:
    """
    Обновляет или добавляет группу пользователей в базу
    """

    stm = insert(models.Group).values(group.model_dump())
    stm = stm.on_conflict_do_update(
        constraint='group_pkey',
        set_={"name": group.name}
    )
    result = await session.execute(stm)

    await session.commit()
    if result:
        return await get_group(session, group.id)
    return None


async def delete_group(
        session: AsyncSession, group_id: int
    ) -> bool:
    """
    Удаляет информацию  о группе пользователей
    """

    has_group = await get_group(session, group_id)
    await session.execute(delete(models.Group) \
                          .filter(models.Group.id == group_id)
                          )
    await session.commit()
    return bool(has_group)
