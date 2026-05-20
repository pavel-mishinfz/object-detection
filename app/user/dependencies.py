from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.database import get_session
from app.user.infrastructure.group_repository import GroupRepository
from app.user.interfaces.repository import IGroupRepository


def get_group_repository(
    session: AsyncSession = Depends(get_session),
) -> IGroupRepository:
    return GroupRepository(session)
