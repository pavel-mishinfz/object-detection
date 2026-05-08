from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.map.domain.errors import PolygonAccessDeniedError, PolygonNotFoundError
from app.map.infrastructure.models import Area


class AreaAccessPolicy:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def check_ownership(self, area_id: UUID, user_id: UUID) -> None:
        owned = await self._session.execute(
            select(Area.id).where(Area.id == area_id, Area.user_id == user_id)
        )
        if owned.scalar_one_or_none() is not None:
            return

        exists = await self._session.execute(
            select(Area.id).where(Area.id == area_id)
        )
        if exists.scalar_one_or_none() is None:
            raise PolygonNotFoundError(f"Полигон {area_id} не найден")
        raise PolygonAccessDeniedError("Нет доступа к данному полигону")
