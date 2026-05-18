from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class ImageInfo:
    id: UUID
    path: str


class IEventPublisher(Protocol):
    async def publish(self, event: object) -> None: ...
    async def run_post_commit(self) -> None: ...


class IAreaAccessPolicy(Protocol):
    async def check_ownership(self, area_id: UUID, user_id: UUID) -> None: ...


class IAreaReader(Protocol):
    async def get_geometry(self, area_id: UUID) -> tuple[tuple[float, float], ...]: ...


class IImageReader(Protocol):
    async def get_images_by_area(self, area_id: UUID) -> list[ImageInfo]: ...
