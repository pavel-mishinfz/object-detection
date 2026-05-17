from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from app.analysis.domain.detection_result import DetectionResult, ObjectType
from app.shared.contracts import IAreaAccessPolicy, ImageInfo, IImageReader  # noqa: F401  (re-export)


@dataclass(frozen=True)
class RawDetection:
    geo_polygon: tuple[tuple[float, float], ...]  # (lon, lat), closed ring
    score: float
    object_type_id: int


class IDetectionResultRepository(ABC):
    @abstractmethod
    async def save(self, result: DetectionResult) -> None:
        pass

    @abstractmethod
    async def find_by_area(self, area_id: UUID) -> list[DetectionResult]:
        pass

    @abstractmethod
    async def delete_by_area(self, area_id: UUID) -> None:
        pass


class IDetectionEngine(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> list[RawDetection]:
        pass


class IObjectTypeRepository(ABC):
    @abstractmethod
    async def find_by_id(self, object_type_id: int) -> ObjectType | None:
        pass

    @abstractmethod
    async def upsert(self, object_type: ObjectType) -> None:
        pass
