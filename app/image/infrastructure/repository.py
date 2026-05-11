from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.image.application.interfaces import IImageRepository
from app.image.domain.image import Image
from app.image.infrastructure.mappers import to_domain
from app.image.infrastructure.models import Image as ImageRecord


class ImageRepository(IImageRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, image: Image) -> None:
        record = ImageRecord(
            id=image.id,
            area_id=image.area_id,
            source=image.source,
            path=image.path,
            bounds_min_lat=image.bounds.min_lat,
            bounds_min_lon=image.bounds.min_lon,
            bounds_max_lat=image.bounds.max_lat,
            bounds_max_lon=image.bounds.max_lon,
            created_at=image.created_at,
        )
        self._session.add(record)
        await self._session.commit()

    async def find_by_id(self, image_id: UUID) -> Image | None:
        result = await self._session.execute(
            select(ImageRecord).where(ImageRecord.id == image_id)
        )
        record = result.scalar_one_or_none()
        return to_domain(record) if record else None

    async def find_by_area(self, area_id: UUID) -> list[Image]:
        result = await self._session.execute(
            select(ImageRecord)
            .where(ImageRecord.area_id == area_id)
            .order_by(ImageRecord.created_at.desc())
        )
        return [to_domain(r) for r in result.scalars().all()]

    async def delete_by_area(self, area_id: UUID) -> None:
        await self._session.execute(
            delete(ImageRecord).where(ImageRecord.area_id == area_id)
        )
        await self._session.commit()
