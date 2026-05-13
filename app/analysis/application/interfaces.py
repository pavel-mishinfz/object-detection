from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.analysis.domain.segmentation_result import ObjectType, SegmentationResult


@dataclass(frozen=True)
class RawContour:
    geo_polygon: tuple[tuple[float, float], ...]  # (lon, lat), closed ring
    object_type_id: int


@dataclass(frozen=True)
class ImageInfo:
    id: UUID
    path: str


class IAreaAccessPolicy(Protocol):
    async def check_ownership(self, area_id: UUID, user_id: UUID) -> None: ...


class IImageReader(Protocol):
    async def get_images_for_area(self, area_id: UUID) -> list[ImageInfo]: ...


class ISegmentationResultRepository(ABC):
    @abstractmethod
    async def save(self, result: SegmentationResult) -> None:
        pass

    @abstractmethod
    async def find_by_area(self, area_id: UUID) -> list[SegmentationResult]:
        pass

    @abstractmethod
    async def delete_by_area(self, area_id: UUID) -> None:
        pass

    @abstractmethod
    async def delete_by_images(self, image_ids: list[UUID]) -> None:
        pass


class ISegmentationEngine(ABC):
    @abstractmethod
    def load(self, model_name: str) -> None:
        pass
    
    @abstractmethod
    def segment(self, image_path: str) -> list[RawContour]:
        pass


class IObjectTypeRepository(ABC):
    @abstractmethod
    async def find_by_id(self, object_type_id: int) -> ObjectType | None:
        pass

    @abstractmethod
    async def upsert(self, object_type: ObjectType) -> None:
        pass
