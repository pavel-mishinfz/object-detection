from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.contracts import IEventPublisher
from app.shared.events import AreaDeleted
from app.map.exceptions import (
    PolygonAccessDeniedError,
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
    crud: AreaRepository,
    session: AsyncSession,
) -> None:
    count = await crud.count_by_user(user_id)
    if count >= 100:
        raise PolygonLimitExceededError(
            "Достигнут лимит полигонов для пользователя (100)"
        )

    await _check_name_exists(user_id, name, crud)

    polygon = build_polygon(
        polygon_id=polygon_id,
        user_id=user_id,
        name=name,
        coordinates=coordinates,
        created_at=datetime.now(),
    )
    await crud.add(polygon)
    await session.commit()


async def update_polygon(
    polygon_id: UUID,
    user_id: UUID,
    name: str,
    coordinates: tuple[tuple[float, float], ...],
    crud: AreaRepository,
    session: AsyncSession,
) -> None:
    existing = await _get_polygon(polygon_id, crud)
    await _check_access(existing.user_id, user_id)
    await _check_name_exists(user_id, name, crud, polygon_id)

    updated = build_updated_polygon(existing, name, coordinates)
    await crud.update(updated)
    await session.commit()


async def delete_polygon(
    polygon_id: UUID,
    user_id: UUID,
    crud: AreaRepository,
    publisher: IEventPublisher,
    session: AsyncSession,
) -> None:
    existing = await _get_polygon(polygon_id, crud)
    await _check_access(existing.user_id, user_id)

    await crud.delete(polygon_id)
    await publisher.publish(AreaDeleted(area_id=polygon_id))
    await session.commit()
    await publisher.run_post_commit()


# --- Impure functions (queries) ---

async def get_polygon(
    polygon_id: UUID,
    user_id: UUID,
    crud: AreaRepository,
) -> Polygon:
    polygon = await _get_polygon(polygon_id, crud)
    await _check_access(polygon.user_id, user_id)
    return polygon


async def get_user_polygons(
    user_id: UUID,
    crud: AreaRepository,
) -> list[Polygon]:
    return await crud.find_by_user(user_id)


async def _get_polygon(polygon_id: UUID, crud: AreaRepository) -> Polygon:
    polygon = await crud.find_by_id(polygon_id)
    if polygon is None:
        raise PolygonNotFoundError(f"Полигон {polygon_id} не найден")
    return polygon


async def _check_name_exists(
    user_id: UUID, name: str, crud: AreaRepository, polygon_id: UUID | None = None
) -> None:
    if await crud.exists_with_name(user_id, name, polygon_id):
        raise PolygonNameConflictError(f"Полигон с именем '{name}' уже существует")


async def _check_access(user_id: UUID, current_user_id: UUID) -> None:
    if user_id != current_user_id:
        raise PolygonAccessDeniedError("Нет доступа к данному полигону")
