from typing import Protocol
from uuid import UUID

from app.shared.dto import ImageInfo


class IImageReader(Protocol):
    async def get_images_by_area(self, area_id: UUID) -> list[ImageInfo]: ...
