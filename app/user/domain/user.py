from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class Group:
    id: int
    name: str


@dataclass(frozen=True)
class User:
    id: UUID
    email: str
    username: str
    group_id: int
    is_active: bool
    is_verified: bool
    is_superuser: bool
