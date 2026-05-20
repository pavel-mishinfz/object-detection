from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.database import get_session
from app.map.infrastructure.adapters import AreaReaderAdapter
from app.map.infrastructure.repository import AreaRepository
from app.map.interfaces.repository import IAreaRepository


def get_area_repository(
    session: AsyncSession = Depends(get_session),
) -> IAreaRepository:
    return AreaRepository(session)


def get_area_reader_adapter(
    session: AsyncSession = Depends(get_session),
) -> AreaReaderAdapter:
    return AreaReaderAdapter(session)
