from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.analysis.domain.detection_result import DetectionResult, ObjectType


@dataclass(frozen=True)
class RawDetection:
    geo_polygon: tuple[tuple[float, float], ...]  # (lon, lat), closed ring
    score: float
    object_type_id: int


@dataclass(frozen=True)
class ImageInfo:
    id: UUID
    path: str


class IAreaAccessPolicy(Protocol):
    async def check_ownership(self, area_id: UUID, user_id: UUID) -> None: ...


class IImageReader(Protocol):
    async def get_images_for_area(self, area_id: UUID) -> list[ImageInfo]: ...


class IDetectionResultRepository(ABC):
    @abstractmethod
    async def save(self, result: DetectionResult) -> DetectionResult:
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
    async def upsert(self, object_type: ObjectType) -> ObjectType:
        pass
