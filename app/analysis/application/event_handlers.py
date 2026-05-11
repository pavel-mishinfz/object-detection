from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.infrastructure.repository import DetectionResultRepository
from app.shared.events import AreaDeleted


async def on_area_deleted(event: AreaDeleted, session: AsyncSession) -> None:
    await DetectionResultRepository(session).delete_by_area(event.area_id)
