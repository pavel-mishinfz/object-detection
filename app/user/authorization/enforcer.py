from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import casbin
from fastapi import Request


@dataclass
class UserSubject:
    id: UUID
    group_id: int


@dataclass
class AreaObject:
    owner_id: UUID
    path: str


def create_enforcer() -> casbin.Enforcer:
    base = Path(__file__).parent
    return casbin.Enforcer(str(base / "model.conf"), str(base / "policy.csv"))


def get_enforcer(request: Request) -> casbin.Enforcer:
    return request.app.state.enforcer
