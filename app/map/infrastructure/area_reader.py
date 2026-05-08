from uuid import UUID

from geoalchemy2.shape import to_shape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.map.infrastructure.models import Area


class AreaReader:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_geometry(self, area_id: UUID) -> tuple[tuple[float, float], ...]:
        result = await self._session.execute(
            select(Area.geometry).where(Area.id == area_id)
        )
        shapely_poly = to_shape(result.scalar_one())
        return tuple((lon, lat) for lon, lat in shapely_poly.coords)
