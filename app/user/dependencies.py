from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.shared.database import get_session
from app.user.infrastructure.auth_backend import get_current_user_id as _get_current_user_id
from app.user.infrastructure.group_repository import GroupRepository


def get_group_repository(
    session: AsyncSession = Depends(get_session),
) -> GroupRepository:
    return GroupRepository(session)


async def get_current_user_id(user_id: UUID = Depends(_get_current_user_id)) -> UUID:
    return user_id
