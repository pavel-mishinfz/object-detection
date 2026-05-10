from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class ImageBounds:
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


@dataclass(frozen=True)
class Image:
    id: UUID
    area_id: UUID
    source: str
    path: str
    bounds: ImageBounds
    created_at: datetime
