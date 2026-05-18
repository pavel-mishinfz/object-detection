from typing import Protocol
from uuid import UUID


class IEventPublisher(Protocol):
    async def publish(self, event: object) -> None: ...
    async def run_post_commit(self) -> None: ...


class IAreaAccessPolicy(Protocol):
    async def check_ownership(self, area_id: UUID, user_id: UUID) -> None: ...
