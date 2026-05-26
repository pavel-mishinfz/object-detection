from abc import ABC, abstractmethod
from uuid import UUID

from app.analysis.entity.detection_result import DetectionResult, ObjectType


class IDetectionResultRepository(ABC):
    @abstractmethod
    async def save(self, result: DetectionResult) -> None: ...
    @abstractmethod
    async def find_by_area(self, area_id: UUID) -> list[DetectionResult]: ...
    @abstractmethod
    async def delete_by_area(self, area_id: UUID) -> None: ...
    @abstractmethod
    async def delete_by_images(self, image_ids: list[UUID]) -> None: ...


class IObjectTypeRepository(ABC):
    @abstractmethod
    async def find_by_id(self, object_type_id: int) -> ObjectType | None: ...
    @abstractmethod
    async def upsert(self, object_type: ObjectType) -> None: ...
