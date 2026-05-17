from uuid import UUID

from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.map.exceptions import PolygonAccessDeniedError, PolygonNotFoundError
from app.map.models.area import Area


class AreaReaderAdapter:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_geometry(self, area_id: UUID) -> tuple[tuple[float, float], ...]:
        result = await self._session.execute(
            select(Area.geometry).where(Area.id == area_id)
        )
        shapely_poly = to_shape(result.scalar_one())
        return tuple((lon, lat) for lon, lat in shapely_poly.exterior.coords)


class AreaAccessPolicyAdapter:
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
