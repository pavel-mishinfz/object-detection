from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.image.application.interfaces import IImageStorage
from app.image.infrastructure.repository import ImageRepository


class ImageFilesCleaner:
    def __init__(self, session: AsyncSession, storage: IImageStorage) -> None:
        self._session = session
        self._storage = storage

    async def delete_files_by_area(self, area_id: UUID) -> None:
        images = await ImageRepository(self._session).find_by_area(area_id)
        for image in images:
            await self._storage.delete(image.path)
