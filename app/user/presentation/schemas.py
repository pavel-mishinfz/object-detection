import uuid
from typing import Optional

from fastapi_users import schemas
from pydantic import BaseModel, Field

from app.user.domain.user import Group


class UserRead(schemas.BaseUser[uuid.UUID]):
    username: str = Field(title="Имя пользователя")
    group_id: int = Field(title="Идентификатор группы")


class UserCreate(schemas.BaseUserCreate):
    username: str
    group_id: int


class UserUpdate(schemas.BaseUserUpdate):
    username: Optional[str] = None
    group_id: Optional[int] = None


class GroupCreate(BaseModel):
    name: str


class GroupRead(BaseModel):
    id: int
    name: str

    model_config = {"from_attributes": True}


class GroupUpdate(BaseModel):
    name: str


def to_group_response(group: Group) -> GroupRead:
    return GroupRead(id=group.id, name=group.name)
