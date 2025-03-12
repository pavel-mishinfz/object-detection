from typing import AsyncGenerator

from fastapi import Depends
from fastapi_users.db import SQLAlchemyUserDatabase

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from . import models


class DatabaseInitializer:
    def __init__(self, base) -> None:
        self.__base = base
        self.__async_session_maker = None

    async def init_database(self, postgres_dsn):
        engine = create_async_engine(postgres_dsn)
        self.__async_session_maker = async_sessionmaker(
            engine, expire_on_commit=False
        )
        async with engine.begin() as conn:
            await conn.run_sync(self.__base.metadata.create_all)

    @property
    def async_session_maker(self):
        return self.__async_session_maker


class Base(DeclarativeBase):
    pass


DB_INITIALIZER = DatabaseInitializer(Base())


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with DB_INITIALIZER.async_session_maker() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, models.User)
