from typing import Protocol
from uuid import UUID


class IAreaReader(Protocol):
    async def get_geometry(self, area_id: UUID) -> tuple[tuple[float, float], ...]: ...
