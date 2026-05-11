from datetime import datetime
from uuid import UUID

from app.map.domain.errors import (
    PolygonAccessDeniedError,
    PolygonLimitExceededError,
    PolygonNameConflictError,
    PolygonNotFoundError,
)
from app.map.domain.polygon import Coordinate, Polygon
from app.map.domain.validators import validate_name, validate_polygon_geometry
from app.map.application.interfaces import IEventPublisher, IPolygonRepository
from app.shared.events import AreaDeleted


# --- Чистые сборщики (нет IO, нет await) ---

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


# --- Команды (impure, -> None) ---

async def create_polygon(
    polygon_id: UUID,
    user_id: UUID,
    name: str,
    coordinates: tuple[tuple[float, float], ...],
    repo: IPolygonRepository,
) -> None:
    count = await repo.count_by_user(user_id)
    if count >= 100:
        raise PolygonLimitExceededError(
            "Достигнут лимит полигонов для пользователя (100)"
        )

    if await repo.exists_with_name(user_id, name):
        raise PolygonNameConflictError(f"Полигон с именем '{name}' уже существует")

    polygon = build_polygon(
        polygon_id=polygon_id,
        user_id=user_id,
        name=name,
        coordinates=coordinates,
        created_at=datetime.now(),
    )
    await repo.save(polygon)


async def update_polygon(
    polygon_id: UUID,
    user_id: UUID,
    name: str,
    coordinates: tuple[tuple[float, float], ...],
    repo: IPolygonRepository,
) -> None:
    existing = await repo.find_by_id(polygon_id)
    if existing is None:
        raise PolygonNotFoundError(f"Полигон {polygon_id} не найден")

    if existing.user_id != user_id:
        raise PolygonAccessDeniedError("Нет доступа к данному полигону")

    if await repo.exists_with_name(user_id, name, polygon_id):
        raise PolygonNameConflictError(f"Полигон с именем '{name}' уже существует")

    updated = build_updated_polygon(existing, name, coordinates)
    await repo.update(updated)


async def delete_polygon(
    polygon_id: UUID,
    user_id: UUID,
    repo: IPolygonRepository,
    publisher: IEventPublisher,
) -> None:
    existing = await repo.find_by_id(polygon_id)
    if existing is None:
        raise PolygonNotFoundError(f"Полигон {polygon_id} не найден")
    if existing.user_id != user_id:
        raise PolygonAccessDeniedError("Нет доступа к данному полигону")
    await repo.delete(polygon_id)
    await publisher.publish(AreaDeleted(area_id=polygon_id))


# --- Запросы (impure, возвращают данные) ---

async def get_polygon(
    polygon_id: UUID,
    user_id: UUID,
    repo: IPolygonRepository,
) -> Polygon:
    polygon = await repo.find_by_id(polygon_id)
    if polygon is None:
        raise PolygonNotFoundError(f"Полигон {polygon_id} не найден")
    if polygon.user_id != user_id:
        raise PolygonAccessDeniedError("Нет доступа к данному полигону")
    return polygon


async def get_user_polygons(
    user_id: UUID,
    repo: IPolygonRepository,
) -> list[Polygon]:
    return await repo.find_by_user(user_id)
