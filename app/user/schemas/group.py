from pydantic import BaseModel

from app.user.entity.group import Group


class GroupBase(BaseModel):
    name: str


class CreateGroupRequest(GroupBase):
    pass


class GroupResponse(GroupBase):
    id: int


class UpdateGroupReqeust(GroupBase):
    pass


def to_group_response(group: Group) -> GroupResponse:
    return GroupResponse(id=group.id, name=group.name)