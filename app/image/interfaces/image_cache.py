from abc import ABC, abstractmethod
from uuid import UUID

from app.image.dto import TilePreview


class IImageCache(ABC):
    @abstractmethod
    async def get(self, area_id: UUID) -> tuple[str, list[TilePreview]] | None: ...
    @abstractmethod
    async def set(self, area_id: UUID, area_hash: str, tiles: list[TilePreview]) -> None: ...
    @abstractmethod
    async def invalidate(self, area_id: UUID) -> None: ...
