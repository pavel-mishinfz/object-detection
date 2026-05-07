from uuid import UUID

from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.image.application.interfaces import IAreaReader
from app.image.domain.errors import AreaAccessDeniedError, AreaNotFoundError
from app.map.infrastructure.models import Area


class AreaReader(IAreaReader):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def check_ownership(self, area_id: UUID, user_id: UUID) -> None:
        await self._assert_ownership(area_id, user_id)

    async def get_geometry(
        self, area_id: UUID, user_id: UUID
    ) -> tuple[tuple[float, float], ...]:
        await self._assert_ownership(area_id, user_id)
        result = await self._session.execute(
            select(Area.geometry).where(Area.id == area_id)
        )
        shapely_poly = to_shape(result.scalar_one())
        return tuple((lon, lat) for lon, lat in shapely_poly.coords)

    async def _assert_ownership(self, area_id: UUID, user_id: UUID) -> None:
        owned = await self._session.execute(
            select(Area.id).where(Area.id == area_id, Area.user_id == user_id)
        )
        if owned.scalar_one_or_none() is not None:
            return

        exists = await self._session.execute(
            select(Area.id).where(Area.id == area_id)
        )
        if exists.scalar_one_or_none() is None:
            raise AreaNotFoundError(f"Полигон {area_id} не найден")
        raise AreaAccessDeniedError("Нет доступа к данному полигону")
