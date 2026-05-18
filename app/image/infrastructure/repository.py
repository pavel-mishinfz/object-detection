from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.image.entity.image import Image
from app.image.mappers import to_domain, to_orm
from app.image.models.image import Image as ImageRecord


class ImageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, image: Image) -> None:
        self._session.add(to_orm(image))
        await self._session.flush()

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
        await self._session.flush()
