from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.contracts import IAreaAccessPolicy, IAreaReader
from app.shared.database import get_session
from app.map.infrastructure.adapters import AreaAccessPolicyAdapter, AreaReaderAdapter
from app.map.infrastructure.repository import AreaRepository


def get_area_repository(
    session: AsyncSession = Depends(get_session),
) -> AreaRepository:
    return AreaRepository(session)


def get_area_reader(
    session: AsyncSession = Depends(get_session),
) -> IAreaReader:
    return AreaReaderAdapter(session)


def get_area_access_policy(
    session: AsyncSession = Depends(get_session),
) -> IAreaAccessPolicy:
    return AreaAccessPolicyAdapter(session)
