from app.user.application.interfaces import IGroupRepository
from app.user.domain.errors import GroupNotFoundError
from app.user.domain.user import Group
from app.user.domain.validators import validate_group_name


# --- Чистые сборщики (нет IO, нет await) ---

def build_updated_group(existing: Group, name: str) -> Group:
    validate_group_name(name)
    return Group(id=existing.id, name=name)


# --- Команды (-> None) ---

async def create_group(name: str, repo: IGroupRepository) -> None:
    validate_group_name(name)
    await repo.create(name)


async def update_group(group_id: int, name: str, repo: IGroupRepository) -> None:
    existing = await repo.find_by_id(group_id)
    if existing is None:
        raise GroupNotFoundError(f"Группа {group_id} не найдена")
    updated = build_updated_group(existing, name)
    await repo.update(updated)


async def delete_group(group_id: int, repo: IGroupRepository) -> None:
    if await repo.find_by_id(group_id) is None:
        raise GroupNotFoundError(f"Группа {group_id} не найдена")
    await repo.delete(group_id)


async def upsert_group(group_id: int, name: str, repo: IGroupRepository) -> None:
    validate_group_name(name)
    await repo.upsert(group_id, name)


# --- Запросы (возвращают данные) ---

async def get_group(group_id: int, repo: IGroupRepository) -> Group:
    group = await repo.find_by_id(group_id)
    if group is None:
        raise GroupNotFoundError(f"Группа {group_id} не найдена")
    return group


async def get_group_by_name(name: str, repo: IGroupRepository) -> Group:
    group = await repo.find_by_name(name)
    if group is None:
        raise GroupNotFoundError(f"Группа '{name}' не найдена")
    return group


async def get_groups(skip: int, limit: int, repo: IGroupRepository) -> list[Group]:
    return await repo.find_all(skip, limit)
