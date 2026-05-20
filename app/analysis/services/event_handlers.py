from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.infrastructure.repository import SegmentationResultRepository
from app.shared.events import AreaDeleted, ImagesByAreaDeleted


async def on_area_deleted(event: AreaDeleted, session: AsyncSession) -> None:
    await SegmentationResultRepository(session).delete_by_area(event.area_id)


async def on_images_deleted(event: ImagesByAreaDeleted, session: AsyncSession) -> None:
    await SegmentationResultRepository(session).delete_by_area(event.area_id)
