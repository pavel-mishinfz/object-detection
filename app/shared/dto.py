from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class ImageInfo:
    id: UUID
    path: str
