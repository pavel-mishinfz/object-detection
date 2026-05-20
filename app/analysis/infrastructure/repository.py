from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.entity.detection_result import DetectionResult, ObjectType
from app.analysis.interfaces.repository import IDetectionResultRepository, IObjectTypeRepository
from app.analysis.mappers import (
    to_domain_detection_result, 
    to_domain_object_type, 
    to_orm_detection_result,
    to_orm_object_type
)
from app.analysis.models.detection_result import DetectionResult as DetectionResultRecord
from app.analysis.models.detection_result import ObjectType as ObjectTypeRecord


class DetectionResultRepository(IDetectionResultRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, result: DetectionResult) -> None:
        self._session.add(to_orm_detection_result(result))
        await self._session.flush()

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
        await self._session.flush()
        
    async def delete_by_images(self, image_ids: list[UUID]) -> None:
        await self._session.execute(
            delete(DetectionResultRecord).where(DetectionResultRecord.image_id.in_(image_ids))
        )
        await self._session.flush()


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
        orm = to_orm_object_type(object_type)
        stmt = (
            pg_insert(ObjectTypeRecord)
            .values(id=orm.id, name=orm.name)
            .on_conflict_do_update(
                index_elements=["id"],
                set_={"name": object_type.name},
            )
        )
        await self._session.execute(stmt)
        await self._session.flush()
