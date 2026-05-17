from dataclasses import dataclass
from uuid import UUID

from app.image.entity.image import ImageBounds


@dataclass(frozen=True)
class TileResult:
    image_id: UUID
    tiff_bytes: bytes


@dataclass(frozen=True)
class PreviewTile:
    image_id: UUID
    bounds: ImageBounds
