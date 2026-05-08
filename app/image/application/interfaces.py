from abc import ABC, abstractmethod
from datetime import date
from typing import Protocol
from uuid import UUID

from app.image.domain.image import Image, PreviewTile, TileResult


class IAreaAccessPolicy(Protocol):
    async def check_ownership(self, area_id: UUID, user_id: UUID) -> None: ...


class IAreaReader(Protocol):
    async def get_geometry(self, area_id: UUID) -> tuple[tuple[float, float], ...]: ...


class IImageRepository(ABC):
    @abstractmethod
    async def save(self, image: Image) -> Image:
        pass

    @abstractmethod
    async def find_by_id(self, image_id: UUID) -> Image | None:
        pass

    @abstractmethod
    async def find_by_area(self, area_id: UUID) -> list[Image]:
        pass

    @abstractmethod
    async def delete_by_area(self, area_id: UUID) -> None:
        pass


class ISentinelGateway(ABC):
    @abstractmethod
    async def fetch_tiles(
        self,
        coordinates: tuple[tuple[float, float], ...],
        date_start: date,
        date_end: date,
    ) -> list[TileResult]:
        pass


class IImageStorage(ABC):
    @abstractmethod
    async def save_temp(self, image_id: UUID, data: bytes) -> None:
        pass

    @abstractmethod
    async def promote_to_permanent(self, image_id: UUID) -> str:
        pass

    @abstractmethod
    async def delete(self, path: str) -> None:
        pass

    @abstractmethod
    async def load_as_png_bytes(self, path: str) -> bytes:
        pass

    @abstractmethod
    def get_temp_path(self, image_id: UUID) -> str:
        pass

    @abstractmethod
    def path_exists(self, path: str) -> bool:
        pass


class IImageCache(ABC):
    @abstractmethod
    async def get(self, area_id: UUID) -> tuple[str, list[PreviewTile]] | None:
        pass

    @abstractmethod
    async def set(
        self,
        area_id: UUID,
        request_hash: str,
        tiles: list[PreviewTile],
    ) -> None:
        pass

    @abstractmethod
    async def invalidate(self, area_id: UUID) -> None:
        pass
