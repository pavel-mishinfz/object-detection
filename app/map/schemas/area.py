import uuid
from datetime import datetime

from pydantic import BaseModel

from app.map.entity.polygon import Polygon


class GeoJSONPolygon(BaseModel):
    type: str = 'Polygon'
    coordinates: list[list[tuple[float, float]]]


class PolygonRequestBase(BaseModel):
    name: str
    geometry: GeoJSONPolygon


class CreatePolygonRequest(PolygonRequestBase):
    pass


class UpdatePolygonRequest(PolygonRequestBase):
    pass


class PolygonResponseBase(BaseModel):
    id: uuid.UUID
    name: str
    created_at: datetime


class PolygonResponse(PolygonResponseBase):
    user_id: uuid.UUID
    geometry: GeoJSONPolygon


class PolygonSummaryResponse(PolygonResponseBase):
    pass


def to_response(polygon: Polygon) -> PolygonResponse:
    return PolygonResponse(
        id=polygon.id,
        user_id=polygon.user_id,
        name=polygon.name,
        created_at=polygon.created_at,
        geometry=GeoJSONPolygon(
            coordinates=[[(c.lon, c.lat) for c in polygon.coordinates]]
        ),
    )


def to_summary_response(polygon: Polygon) -> PolygonSummaryResponse:
    return PolygonSummaryResponse(
        id=polygon.id,
        name=polygon.name,
        created_at=polygon.created_at,
    )
