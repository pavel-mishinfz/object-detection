from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class Coordinate:
    lat: float  # [-90, 90]
    lon: float  # [-180, 180]


@dataclass(frozen=True)
class Polygon:
    id: UUID
    user_id: UUID
    name: str
    coordinates: tuple[Coordinate, ...]
    created_at: datetime
