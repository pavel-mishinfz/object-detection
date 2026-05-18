from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.contracts import IEventPublisher
from app.shared.events import AreaDeleted
from app.map.exceptions import (
    PolygonLimitExceededError,
    PolygonNameConflictError,
    PolygonNotFoundError,
)
from app.map.entity.polygon import Coordinate, Polygon
from app.map.infrastructure.repository import AreaRepository
from app.map.services.validators import validate_name, validate_polygon_geometry


# --- Pure functions ---

def build_polygon(
    polygon_id: UUID,
    user_id: UUID,
    name: str,
    coordinates: tuple[tuple[float, float], ...],
    created_at: datetime,
) -> Polygon:
    validate_name(name)
    coords = _to_coordinates(coordinates)
    validate_polygon_geometry(coords)
    return Polygon(
        id=polygon_id,
        user_id=user_id,
        name=name,
        coordinates=coords,
        created_at=created_at,
    )


def build_updated_polygon(
    existing: Polygon,
    name: str,
    coordinates: tuple[tuple[float, float], ...],
) -> Polygon:
    validate_name(name)
    coords = _to_coordinates(coordinates)
    validate_polygon_geometry(coords)
    return Polygon(
        id=existing.id,
        user_id=existing.user_id,
        name=name,
        coordinates=coords,
        created_at=existing.created_at,
    )


def _to_coordinates(raw: tuple[tuple[float, float], ...]) -> tuple[Coordinate, ...]:
    return tuple(Coordinate(lat=lat, lon=lon) for lat, lon in raw)


# --- Impure functions (commands) ---

async def create_polygon(
    polygon_id: UUID,
    user_id: UUID,
    name: str,
    coordinates: tuple[tuple[float, float], ...],
    repo: AreaRepository,
    session: AsyncSession,
) -> None:
    count = await repo.count_by_user(user_id)
    if count >= 100:
        raise PolygonLimitExceededError(
            "Достигнут лимит полигонов для пользователя (100)"
        )

    await _check_name_exists(user_id, name, repo)

    polygon = build_polygon(
        polygon_id=polygon_id,
        user_id=user_id,
        name=name,
        coordinates=coordinates,
        created_at=datetime.now(),
    )
    await repo.add(polygon)
    await session.commit()


async def update_polygon(
    polygon_id: UUID,
    user_id: UUID,
    name: str,
    coordinates: tuple[tuple[float, float], ...],
    repo: AreaRepository,
    session: AsyncSession,
) -> None:
    existing = await _get_polygon(polygon_id, repo)
    await _check_name_exists(user_id, name, repo, polygon_id)
    updated = build_updated_polygon(existing, name, coordinates)
    await repo.update(updated)
    await session.commit()


async def delete_polygon(
    polygon_id: UUID,
    repo: AreaRepository,
    publisher: IEventPublisher,
    session: AsyncSession,
) -> None:
    await repo.delete(polygon_id)
    await publisher.publish(AreaDeleted(area_id=polygon_id))
    await session.commit()
    await publisher.run_post_commit()


# --- Impure functions (queries) ---

async def get_polygon(polygon_id: UUID, repo: AreaRepository) -> Polygon:
    return await _get_polygon(polygon_id, repo)


async def get_user_polygons(
    user_id: UUID,
    repo: AreaRepository,
) -> list[Polygon]:
    return await repo.find_by_user(user_id)


async def _get_polygon(polygon_id: UUID, repo: AreaRepository) -> Polygon:
    polygon = await repo.find_by_id(polygon_id)
    if polygon is None:
        raise PolygonNotFoundError(f"Полигон {polygon_id} не найден")
    return polygon


async def _check_name_exists(
    user_id: UUID, name: str, repo: AreaRepository, polygon_id: UUID | None = None
) -> None:
    if await repo.exists_with_name(user_id, name, polygon_id):
        raise PolygonNameConflictError(f"Полигон с именем '{name}' уже существует")
