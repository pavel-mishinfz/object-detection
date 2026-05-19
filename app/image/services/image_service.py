import hashlib
from datetime import date, datetime
import json
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.image.dto import TilePreview
from app.image.entity.image import Image
from app.image.exceptions import ImageNotFoundError, NoPreviewAvailableError
from app.image.infrastructure.image_storage import LocalImageStorage
from app.image.infrastructure.repository import ImageRepository
from app.image.infrastructure.sentinel_gateway import SentinelHubGateway
from app.image.infrastructure.image_cache import RedisImageCache
from app.image.services.validators import validate_date_range
from app.image.contracts import IAreaReader
from app.shared.contracts import IEventPublisher
from app.shared.events import ImagesByAreaDeleted


# --- Pure functions ---

def compute_area_hash(
    coordinates: tuple[tuple[float, float], ...],
    date_start: date,
    date_end: date,
) -> str:
    data = {
        "coordinates": sorted(coordinates),
        "date_start": date_start.isoformat(),
        "date_end": date_end.isoformat(),
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


# --- Impure functions (commands) ---

async def fetch_previews(
    area_id: UUID,
    date_start: date,
    date_end: date,
    area_reader: IAreaReader,
    sentinel_gateway: SentinelHubGateway,
    cache: RedisImageCache,
    storage: LocalImageStorage,
) -> None:
    validate_date_range(date_start, date_end, date.today())
    coordinates = await area_reader.get_geometry(area_id)

    area_hash = compute_area_hash(coordinates, date_start, date_end)

    cached = await cache.get(area_id)
    if cached is not None:
        cached_hash, _ = cached
        if cached_hash == area_hash:
            return

    preview_tiles: list[TilePreview] = []
    tiles = await sentinel_gateway.fetch_tiles(coordinates, date_start, date_end)

    for tile in tiles:
        await storage.save_temp(tile.image_id, tile.tiff_bytes)
        preview_tiles.append(TilePreview(image_id=tile.image_id, bounds=tile.bounds))

    await cache.set(area_id, area_hash, preview_tiles)


async def save_images(
    area_id: UUID,
    repo: ImageRepository,
    cache: RedisImageCache,
    storage: LocalImageStorage,
    session: AsyncSession,
) -> None:
    cached = await cache.get(area_id)
    if cached is None:
        raise NoPreviewAvailableError(
            "Нет данных предпросмотра для сохранения. Выполните предпросмотр снимков."
        )

    _, tiles = cached
    now = datetime.now()
    for tile in tiles:
        await storage.move_to_permanent_storage(tile.image_id)
        path = await storage.get_permanent_path(tile.image_id)
        image = Image(
            id=tile.image_id,
            area_id=area_id,
            source="SENTINEL2_L2A",
            path=path,
            bounds=tile.bounds,
            created_at=now,
        )
        await repo.save(image)
    await session.commit()

    await cache.invalidate(area_id)


async def delete_images(
    area_id: UUID,
    repo: ImageRepository,
    storage: LocalImageStorage,
    publisher: IEventPublisher,
    session: AsyncSession,
) -> None:
    images = await repo.find_by_area(area_id)
    await repo.delete_by_area(area_id)
    await publisher.publish(ImagesByAreaDeleted(area_id=area_id))
    await session.commit()
    await publisher.run_post_commit()
    for image in images:
        await storage.delete(image.path)


# --- Impure functions (queries) ---

async def get_preview_tiles(
    area_id: UUID,
    cache: RedisImageCache,
) -> list[TilePreview]:
    cached = await cache.get(area_id)
    if cached is None:
        return []
    _, tiles = cached
    return tiles


async def get_images(
    area_id: UUID,
    repo: ImageRepository,
) -> list[Image]:
    return await repo.find_by_area(area_id)


async def get_image_as_png(
    image_id: UUID,
    repo: ImageRepository,
    storage: LocalImageStorage,
) -> bytes:
    image = await repo.find_by_id(image_id)
    if image is not None:
        return await storage.get_image_as_png_bytes(image.path)

    temp_path = storage.get_temp_path(image_id)
    if temp_path is not None:
        return await storage.get_image_as_png_bytes(temp_path)

    raise ImageNotFoundError(f"Снимок {image_id} не найден")
