import uuid
from typing import Optional

from fastapi_users import schemas
from pydantic import Field


class UserRead(schemas.BaseUser[uuid.UUID]):
    username: str = Field(title='Имя пользователя')
    group_id: int = Field(title='Индентификатор группы')


class UserCreate(schemas.BaseUserCreate):
    username: str
    group_id: int


class UserUpdate(schemas.BaseUserUpdate):
    username: Optional[str]
    group_id: Optional[int]
