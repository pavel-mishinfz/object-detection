from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.map.entity.polygon import Polygon
from app.map.mappers import to_domain, to_orm
from app.map.models.area import Area


class AreaRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, polygon: Polygon) -> None:
        self._session.add(to_orm(polygon))
        await self._session.flush()

    async def find_by_id(self, polygon_id: UUID) -> Polygon | None:
        result = await self._session.execute(
            select(Area).where(Area.id == polygon_id)
        )
        record = result.scalar_one_or_none()
        return to_domain(record) if record else None

    async def find_by_user(self, user_id: UUID) -> list[Polygon]:
        result = await self._session.execute(
            select(Area)
            .where(Area.user_id == user_id)
            .order_by(Area.created_at.desc())
        )
        return [to_domain(r) for r in result.scalars().all()]

    async def update(self, polygon: Polygon) -> None:
        orm = to_orm(polygon)
        await self._session.execute(
            update(Area)
            .where(Area.id == polygon.id)
            .values(name=orm.name, geometry=orm.geometry)
        )
        await self._session.flush()

    async def delete(self, polygon_id: UUID) -> None:
        await self._session.execute(
            delete(Area).where(Area.id == polygon_id)
        )
        await self._session.flush()

    async def exists_with_name(
        self, user_id: UUID, name: str, exclude_id: UUID | None = None
    ) -> bool:
        stmt = select(Area.id).where(
            Area.user_id == user_id,
            Area.name == name,
        )
        if exclude_id is not None:
            stmt = stmt.where(Area.id != exclude_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def count_by_user(self, user_id: UUID) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(Area).where(Area.user_id == user_id)
        )
        return result.scalar_one()
