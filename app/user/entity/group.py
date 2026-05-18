from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    id: int
    name: str
