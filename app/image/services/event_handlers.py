from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.image.infrastructure.image_storage import LocalImageStorage
from app.image.infrastructure.repository import ImageRepository
from app.shared.events import AreaDeleted


def make_on_area_deleted(storage: LocalImageStorage) -> Callable:
    async def on_area_deleted(event: AreaDeleted, session: AsyncSession) -> list[Callable]:
        repo = ImageRepository(session)
        images = await repo.find_by_area(event.area_id)
        await repo.delete_by_area(event.area_id)
        return [lambda p=img.path: storage.delete(p) for img in images]
    return on_area_deleted
