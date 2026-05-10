from abc import ABC, abstractmethod
from typing import List, Protocol
from uuid import UUID

from app.map.domain.polygon import Polygon


class IImageFilesCleaner(Protocol):
    async def delete_files_by_area(self, area_id: UUID) -> None: ...


class IPolygonRepository(ABC):
    @abstractmethod
    async def save(self, polygon: Polygon) -> None:
        """Сохраняет полигон в БД"""
        pass

    @abstractmethod
    async def find_by_id(self, polygon_id: UUID) -> Polygon | None:
        """Находит полигон по ID."""
        pass

    @abstractmethod
    async def find_by_user(self, user_id: UUID) -> List[Polygon]:
        """Находит все полигоны пользователя"""
        pass

    @abstractmethod
    async def update(self, polygon: Polygon) -> None:
        """Обновляет существующий полигон"""
        pass

    @abstractmethod
    async def delete(self, polygon_id: UUID) -> None:
        """Удаляет полигон по ID"""
        pass

    @abstractmethod
    async def exists_with_name(
            self, user_id: UUID, name: str, exclude_id: UUID | None = None
    ) -> bool:
        """Проверяет существование полигона с таким именем у пользователя"""
        pass

    @abstractmethod
    async def count_by_user(self, user_id: UUID) -> int:
        """Возвращает количество полигонов пользователя"""
        pass
