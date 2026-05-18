from sqlalchemy.ext.asyncio import AsyncSession

from app.user.entity.group import Group
from app.user.exceptions import GroupNotFoundError
from app.user.infrastructure.group_repository import GroupRepository
from app.user.services.validators import validate_group_name


# --- Pure functions ---

def build_updated_group(existing: Group, name: str) -> Group:
    validate_group_name(name)
    return Group(id=existing.id, name=name)


# --- Commands ---

async def create_group(name: str, repo: GroupRepository, session: AsyncSession) -> None:
    validate_group_name(name)
    await repo.save(name)
    await session.commit()


async def update_group(
    group_id: int, name: str, repo: GroupRepository, session: AsyncSession
) -> None:
    existing = await repo.find_by_id(group_id)
    if existing is None:
        raise GroupNotFoundError(f"Группа {group_id} не найдена")
    updated = build_updated_group(existing, name)
    await repo.update(updated)
    await session.commit()


async def delete_group(
    group_id: int, repo: GroupRepository, session: AsyncSession
) -> None:
    if await repo.find_by_id(group_id) is None:
        raise GroupNotFoundError(f"Группа {group_id} не найдена")
    await repo.delete(group_id)
    await session.commit()


async def upsert_group(
    group_id: int, name: str, repo: GroupRepository, session: AsyncSession
) -> None:
    validate_group_name(name)
    await repo.upsert(group_id, name)
    await session.commit()


# --- Queries ---

async def get_group(group_id: int, repo: GroupRepository) -> Group:
    group = await repo.find_by_id(group_id)
    if group is None:
        raise GroupNotFoundError(f"Группа {group_id} не найдена")
    return group


async def get_group_by_name(name: str, repo: GroupRepository) -> Group:
    group = await repo.find_by_name(name)
    if group is None:
        raise GroupNotFoundError(f"Группа {name} не найдена")
    return group


async def get_groups(skip: int, limit: int, repo: GroupRepository) -> list[Group]:
    return await repo.find_all(skip, limit)
