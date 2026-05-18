from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.database import get_session
from app.user.application.interfaces import IGroupRepository
from app.user.infrastructure.group_repository import GroupRepository


__all__ = ["get_group_repository"]


def get_group_repository(
    session: AsyncSession = Depends(get_session),
) -> IGroupRepository:
    return GroupRepository(session)
