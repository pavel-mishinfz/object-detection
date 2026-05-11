from uuid import UUID

from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.application.interfaces import IDetectionResultRepository, IObjectTypeRepository
from app.analysis.domain.detection_result import DetectionResult, ObjectType
from app.analysis.infrastructure.mappers import to_domain_detection_result, to_domain_object_type
from app.analysis.infrastructure.models import DetectionResult as DetectionResultRecord
from app.analysis.infrastructure.models import ObjectType as ObjectTypeRecord


class DetectionResultRepository(IDetectionResultRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, result: DetectionResult) -> None:
        record = DetectionResultRecord(
            id=result.id,
            area_id=result.area_id,
            image_id=result.image_id,
            geometry=from_shape(Polygon(result.geometry), srid=4326),
            score=result.score,
            object_type_id=result.object_type.id,
            created_at=result.created_at,
        )
        self._session.add(record)
        await self._session.commit()

    async def find_by_area(self, area_id: UUID) -> list[DetectionResult]:
        result = await self._session.execute(
            select(DetectionResultRecord)
            .where(DetectionResultRecord.area_id == area_id)
            .order_by(DetectionResultRecord.created_at.desc())
        )
        return [to_domain_detection_result(r) for r in result.scalars().all()]

    async def delete_by_area(self, area_id: UUID) -> None:
        await self._session.execute(
            delete(DetectionResultRecord).where(DetectionResultRecord.area_id == area_id)
        )
        await self._session.commit()


class ObjectTypeRepository(IObjectTypeRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_id(self, object_type_id: int) -> ObjectType | None:
        result = await self._session.execute(
            select(ObjectTypeRecord).where(ObjectTypeRecord.id == object_type_id)
        )
        record = result.scalar_one_or_none()
        return to_domain_object_type(record) if record else None

    async def upsert(self, object_type: ObjectType) -> None:
        stmt = (
            pg_insert(ObjectTypeRecord)
            .values(id=object_type.id, name=object_type.name)
            .on_conflict_do_update(
                index_elements=["id"],
                set_={"name": object_type.name},
            )
        )
        await self._session.execute(stmt)
        await self._session.commit()
