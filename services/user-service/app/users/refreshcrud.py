import typing
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from .database import models
from . import schemas


async def create_refresh_token(
        refresh_token: schemas.refresh_token.RefreshTokenIn, session: AsyncSession
    ) -> models.RefreshToken:
    """
    Создает новый refresh-токен пользователя в базе
    """
    
    db_refresh_token = models.RefreshToken(**refresh_token.model_dump())

    session.add(db_refresh_token)
    await session.commit()
    await session.refresh(db_refresh_token)
    return db_refresh_token

async def get_refresh_token(
        session: AsyncSession, refresh_token: str
    ) -> models.RefreshToken | None:
    """
    Возвращает информацию о refresh токене
    """

    result = await session.execute(select(models.RefreshToken) \
                                   .where(models.RefreshToken.refresh_token == refresh_token) \
                                   .limit(1)
                                   )
    return result.scalars().one_or_none()

async def get_refresh_token_by_user_id(
        session: AsyncSession, user_id: str
    ) -> models.RefreshToken | None:
    """
    Возвращает информацию о refresh токене по id пользователя
    """

    result = await session.execute(select(models.RefreshToken) \
                                   .where(models.RefreshToken.user_id == user_id) \
                                   .limit(1)
                                   )
    return result.scalars().one_or_none()

async def delete_refresh_token(
        session: AsyncSession, refresh_token: str
    ) -> bool:
    """
    Удаляет информацию  о refresh токене
    """

    has_refresh_token = await get_refresh_token(session, refresh_token)
    await session.execute(delete(models.RefreshToken) \
                          .filter(models.RefreshToken.refresh_token == refresh_token)
                          )
    await session.commit()
    return bool(has_refresh_token)