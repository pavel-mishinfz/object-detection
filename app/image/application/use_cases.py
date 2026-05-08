import hashlib
import json
from datetime import date, datetime
from uuid import UUID

from app.image.domain.errors import (
    ImageNotFoundError,
    InvalidDateRangeError,
    NoPreviewAvailableError,
)
from app.image.domain.image import Image, ImageBounds, PreviewTile
from app.image.application.interfaces import (
    IAreaAccessPolicy,
    IAreaReader,
    IImageCache,
    IImageRepository,
    IImageStorage,
    ISentinelGateway,
)


# --- Чистые функции (нет IO) ---

def compute_request_hash(
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


def validate_date_range(date_start: date, date_end: date) -> None:
    today = date.today()
    if date_start > today or date_end > today:
        raise InvalidDateRangeError("Дата не может быть в будущем")
    if date_start > date_end:
        raise InvalidDateRangeError(
            "Дата начала должна быть раньше или равна дате окончания"
        )


def build_image(
    image_id: UUID,
    area_id: UUID,
    path: str,
    bounds: ImageBounds,
    created_at: datetime,
) -> Image:
    return Image(
        id=image_id,
        area_id=area_id,
        source="SENTINEL2_L2A",
        path=path,
        bounds=bounds,
        created_at=created_at,
    )


# --- Команды (impure, -> None) ---

async def fetch_previews(
    area_id: UUID,
    user_id: UUID,
    date_start: date,
    date_end: date,
    area_access_policy: IAreaAccessPolicy,
    area_reader: IAreaReader,
    gateway: ISentinelGateway,
    cache: IImageCache,
    storage: IImageStorage,
) -> None:
    validate_date_range(date_start, date_end)
    await area_access_policy.check_ownership(area_id, user_id)
    coordinates = await area_reader.get_geometry(area_id)
    request_hash = compute_request_hash(coordinates, date_start, date_end)

    cached = await cache.get(area_id)
    if cached is not None:
        cached_hash, _ = cached
        if cached_hash == request_hash:
            return

    tile_results = await gateway.fetch_tiles(coordinates, date_start, date_end)

    tiles: list[PreviewTile] = []
    for result in tile_results:
        await storage.save_temp(result.image_id, result.tiff_bytes)
        tiles.append(PreviewTile(image_id=result.image_id, bounds=result.bounds))

    await cache.set(area_id, request_hash, tiles)


async def save_images(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: IImageRepository,
    storage: IImageStorage,
    cache: IImageCache,
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)

    cached = await cache.get(area_id)
    if cached is None:
        raise NoPreviewAvailableError(
            "Нет данных предпросмотра для сохранения. Выполните предпросмотр снимков."
        )

    _, tiles = cached
    now = datetime.now()

    for tile in tiles:
        path = await storage.promote_to_permanent(tile.image_id)
        image = build_image(
            image_id=tile.image_id,
            area_id=area_id,
            path=path,
            bounds=tile.bounds,
            created_at=now,
        )
        await repo.save(image)

    await cache.invalidate(area_id)


async def delete_images(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: IImageRepository,
    storage: IImageStorage,
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)
    images = await repo.find_by_area(area_id)
    for image in images:
        await storage.delete(image.path)
    await repo.delete_by_area(area_id)


# --- Запросы (impure, возвращают данные) ---

async def get_preview_tiles(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    cache: IImageCache,
) -> list[PreviewTile]:
    await area_access_policy.check_ownership(area_id, user_id)
    cached = await cache.get(area_id)
    if cached is None:
        return []
    _, tiles = cached
    return tiles


async def get_images(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: IImageRepository,
) -> list[Image]:
    await area_access_policy.check_ownership(area_id, user_id)
    return await repo.find_by_area(area_id)


async def get_image_as_png(
    image_id: UUID,
    user_id: UUID,
    repo: IImageRepository,
    area_access_policy: IAreaAccessPolicy,
    storage: IImageStorage,
) -> bytes:
    image = await repo.find_by_id(image_id)
    if image is not None:
        await area_access_policy.check_ownership(image.area_id, user_id)
        return await storage.load_as_png_bytes(image.path)

    temp_path = storage.get_temp_path(image_id)
    if storage.path_exists(temp_path):
        return await storage.load_as_png_bytes(temp_path)

    raise ImageNotFoundError(f"Снимок {image_id} не найден")
