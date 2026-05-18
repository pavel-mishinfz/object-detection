import uuid
from typing import Optional

from fastapi_users import schemas


class UserRead(schemas.BaseUser[uuid.UUID]):
    username: str
    group_id: int


class UserCreate(schemas.BaseUserCreate):
    username: str
    group_id: int


class UserUpdate(schemas.BaseUserUpdate):
    username: Optional[str] = None
    group_id: Optional[int] = None
