from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class ObjectType:
    id: int
    name: str


@dataclass(frozen=True)
class SegmentationResult:
    id: UUID
    area_id: UUID
    image_id: UUID
    geometry: tuple[tuple[float, float], ...]
    object_type: ObjectType
    created_at: datetime
