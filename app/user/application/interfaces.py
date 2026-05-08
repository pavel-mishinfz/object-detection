from abc import ABC, abstractmethod

from app.user.domain.user import Group


class IGroupRepository(ABC):
    @abstractmethod
    async def create(self, name: str) -> None:
        """Создаёт новую группу"""
        pass

    @abstractmethod
    async def find_by_id(self, group_id: int) -> Group | None:
        """Находит группу по ID"""
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> Group | None:
        """Находит группу по имени"""
        pass

    @abstractmethod
    async def find_all(self, skip: int, limit: int) -> list[Group]:
        """Возвращает все группы с пагинацией"""
        pass

    @abstractmethod
    async def update(self, group: Group) -> None:
        """Обновляет группу"""
        pass

    @abstractmethod
    async def delete(self, group_id: int) -> None:
        """Удаляет группу по ID"""
        pass

    @abstractmethod
    async def upsert(self, group_id: int, name: str) -> None:
        """Вставляет или обновляет группу по ID (для seed-данных)"""
        pass


class IEmailSender(ABC):
    @abstractmethod
    async def send_reset_password(self, to_email: str, token: str) -> None:
        """Отправляет письмо для сброса пароля"""
        pass

    @abstractmethod
    async def send_verification(self, to_email: str, token: str) -> None:
        """Отправляет письмо для подтверждения аккаунта"""
        pass
