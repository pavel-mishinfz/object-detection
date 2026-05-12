from datetime import date, datetime
from uuid import UUID

from app.image.domain.errors import (
    ImageNotFoundError,
    NoPreviewAvailableError,
)
from app.image.domain.image import Image, ImageBounds
from app.image.application.interfaces import (
    IAreaAccessPolicy,
    IAreaReader,
    IEventPublisher,
    IImageRepository,
    IImageStorage,
    ISentinelGateway,
    PreviewTile,
)
from app.image.domain.validators import validate_date_range
from app.shared.events import ImagesDeleted


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
    storage: IImageStorage,
) -> None:
    validate_date_range(date_start, date_end, date.today())
    await area_access_policy.check_ownership(area_id, user_id)
    coordinates = await area_reader.get_geometry(area_id)

    tile_results = await gateway.fetch_tiles(coordinates, date_start, date_end)

    for result in tile_results:
        await storage.save_temp(area_id, result.image_id, result.tiff_bytes)


async def delete_previews(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    storage: IImageStorage,
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)

    temp_tile_ids = await storage.list_temp_by_area(area_id)
    for tile_id in temp_tile_ids:
        tile_path = storage.find_temp_path(tile_id)
        await storage.delete(tile_path)


async def save_images(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: IImageRepository,
    storage: IImageStorage,
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)

    image_ids = await storage.list_temp_by_area(area_id)
    if not image_ids:
        raise NoPreviewAvailableError(
            "Нет данных предпросмотра для сохранения. Выполните предпросмотр снимков."
        )

    now = datetime.now()
    for image_id in image_ids:
        bounds = await storage.get_temp_bounds(image_id)
        path = await storage.promote_to_permanent(image_id)
        image = build_image(
            image_id=image_id,
            area_id=area_id,
            path=path,
            bounds=bounds,
            created_at=now,
        )
        await repo.save(image)


async def delete_images(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    repo: IImageRepository,
    storage: IImageStorage,
    publisher: IEventPublisher
) -> None:
    await area_access_policy.check_ownership(area_id, user_id)
    images = await repo.find_by_area(area_id)
    await repo.delete_by_area(area_id)
    await publisher.publish(ImagesDeleted(image_ids=[img.id for img in images]))
    for image in images:
        await storage.delete(image.path)


# --- Запросы (impure, возвращают данные) ---

async def get_preview_tiles(
    area_id: UUID,
    user_id: UUID,
    area_access_policy: IAreaAccessPolicy,
    storage: IImageStorage,
) -> list[PreviewTile]:
    await area_access_policy.check_ownership(area_id, user_id)
    
    tiles: list[PreviewTile] = []
    image_ids = await storage.list_temp_by_area(area_id)
    
    for image_id in image_ids:
        bounds = await storage.get_temp_bounds(image_id)
        tiles.append(PreviewTile(image_id, bounds))
    
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

    temp_path = storage.find_temp_path(image_id)
    if temp_path is not None:
        return await storage.load_as_png_bytes(temp_path)

    raise ImageNotFoundError(f"Снимок {image_id} не найден")
