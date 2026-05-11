from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import load_config
from app.image.infrastructure.image_storage import FileImageStorage
from app.image.infrastructure.repository import ImageRepository
from app.shared.events import AreaDeleted

_cfg = load_config()


async def on_area_deleted(event: AreaDeleted, session: AsyncSession) -> list[Callable]:
    storage = FileImageStorage(_cfg.sentinel_temp_dir, _cfg.sentinel_images_dir)
    repo = ImageRepository(session)
    images = await repo.find_by_area(event.area_id)
    await repo.delete_by_area(event.area_id)
    return [lambda p=img.path: storage.delete(p) for img in images]
