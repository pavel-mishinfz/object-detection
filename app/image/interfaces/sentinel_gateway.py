from abc import ABC, abstractmethod
from datetime import date

from app.image.dto import TileResult


class ISentinelGateway(ABC):
    @abstractmethod
    async def fetch_tiles(
        self,
        coordinates: tuple[tuple[float, float], ...],
        date_start: date,
        date_end: date,
    ) -> list[TileResult]: ...
