from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class AreaDeleted:
    area_id: UUID


@dataclass(frozen=True)
class ImagesDeleted:
    image_ids: list[UUID]