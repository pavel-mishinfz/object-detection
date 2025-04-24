import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from ..database import models

from geoalchemy2 import WKBElement


async def create_analysis(
        db: AsyncSession, image_id: uuid.UUID, geometry: WKBElement, object_type_id: int
    ) -> models.SegmentationResult | None:
    """
    Сохраняет информацию о результате сегментирования снимка
    """

    db_analysis = models.SegmentationResult(
        image_id=image_id,
        geometry=geometry,
        object_type_id=object_type_id
    )

    db.add(db_analysis)
    await db.commit()
    await db.refresh(db_analysis)

    return db_analysis
