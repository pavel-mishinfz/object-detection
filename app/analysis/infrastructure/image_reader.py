from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.application.interfaces import IImageReader, ImageInfo
from app.image.infrastructure.models import Image


class ImageReader(IImageReader):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_images_for_area(self, area_id: UUID) -> list[ImageInfo]:
        result = await self._session.execute(
            select(Image).where(Image.area_id == area_id)
        )
        return [ImageInfo(id=r.id, path=r.path) for r in result.scalars().all()]
