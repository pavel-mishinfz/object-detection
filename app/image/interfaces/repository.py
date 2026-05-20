from abc import ABC, abstractmethod
from uuid import UUID

from app.image.entity.image import Image


class IImageRepository(ABC):
    @abstractmethod
    async def save(self, image: Image) -> None: ...
    @abstractmethod
    async def find_by_id(self, image_id: UUID) -> Image | None: ...
    @abstractmethod
    async def find_by_area(self, area_id: UUID) -> list[Image]: ...
    @abstractmethod
    async def delete_by_area(self, area_id: UUID) -> None: ...
