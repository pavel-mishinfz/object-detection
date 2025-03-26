import uuid
import json
from typing import List, Tuple

from sqlalchemy import delete, select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from .database import models


async def get_coordinates_by_area_id(
        db: AsyncSession, area_id: uuid.UUID
) -> bytes | None:
    """
    Возвращает координаты полигона в виде WKB
    """
    query = text("""
        SELECT geometry
        FROM area
        WHERE id = :area_id
    """)
    result = await db.execute(query, {"area_id": str(area_id)})
    row = result.fetchone()

    if not row:
        return None

    wkb_data = row[0]
    return wkb_data
