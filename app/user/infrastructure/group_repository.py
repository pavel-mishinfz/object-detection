from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.user.entity.group import Group
from app.user.mappers import to_domain_group
from app.user.models.group import Group as GroupRecord


class GroupRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, name: str) -> None:
        record = GroupRecord(name=name)
        self._session.add(record)
        await self._session.flush()

    async def find_by_id(self, group_id: int) -> Group | None:
        result = await self._session.execute(
            select(GroupRecord).where(GroupRecord.id == group_id)
        )
        record = result.scalar_one_or_none()
        return to_domain_group(record) if record else None

    async def find_by_name(self, name: str) -> Group | None:
        result = await self._session.execute(
            select(GroupRecord).where(GroupRecord.name == name)
        )
        record = result.scalar_one_or_none()
        return to_domain_group(record) if record else None

    async def find_all(self, skip: int, limit: int) -> list[Group]:
        result = await self._session.execute(
            select(GroupRecord).offset(skip).limit(limit)
        )
        return [to_domain_group(r) for r in result.scalars().all()]

    async def update(self, group: Group) -> None:
        await self._session.execute(
            update(GroupRecord)
            .where(GroupRecord.id == group.id)
            .values(name=group.name)
        )

    async def delete(self, group_id: int) -> None:
        await self._session.execute(
            delete(GroupRecord).where(GroupRecord.id == group_id)
        )

    async def upsert(self, group_id: int, name: str) -> None:
        stmt = pg_insert(GroupRecord).values(id=group_id, name=name)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={"name": name},
        )
        await self._session.execute(stmt)
