from dataclasses import dataclass
from uuid import UUID

from app.image.entity.image import ImageBounds


@dataclass(frozen=True)
class TileResult:
    image_id: UUID
    bounds: ImageBounds
    tiff_bytes: bytes


@dataclass(frozen=True)
class TilePreview:
    image_id: UUID
    bounds: ImageBounds
