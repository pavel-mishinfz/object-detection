from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class AreaDeleted:
    area_id: UUID


@dataclass(frozen=True)
class ImagesByAreaDeleted:
    area_id: UUID
