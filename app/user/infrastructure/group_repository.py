from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.user.application.interfaces import IGroupRepository
from app.user.domain.user import Group
from app.user.infrastructure.mappers import to_domain_group
from app.user.infrastructure.models import Group as GroupModel


class GroupRepository(IGroupRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, name: str) -> None:
        record = GroupModel(name=name)
        self._session.add(record)
        await self._session.flush()

    async def find_by_id(self, group_id: int) -> Group | None:
        result = await self._session.execute(
            select(GroupModel).where(GroupModel.id == group_id)
        )
        record = result.scalar_one_or_none()
        return to_domain_group(record) if record else None

    async def find_by_name(self, name: str) -> Group | None:
        result = await self._session.execute(
            select(GroupModel).where(GroupModel.name == name)
        )
        record = result.scalar_one_or_none()
        return to_domain_group(record) if record else None

    async def find_all(self, skip: int, limit: int) -> list[Group]:
        result = await self._session.execute(
            select(GroupModel).offset(skip).limit(limit)
        )
        return [to_domain_group(r) for r in result.scalars().all()]

    async def update(self, group: Group) -> None:
        await self._session.execute(
            update(GroupModel)
            .where(GroupModel.id == group.id)
            .values(name=group.name)
        )
        await self._session.flush()

    async def delete(self, group_id: int) -> None:
        await self._session.execute(
            delete(GroupModel).where(GroupModel.id == group_id)
        )
        await self._session.flush()

    async def upsert(self, group_id: int, name: str) -> None:
        stmt = pg_insert(GroupModel).values(id=group_id, name=name)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={"name": name},
        )
        await self._session.execute(stmt)
        await self._session.flush()
