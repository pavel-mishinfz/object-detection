import uuid
from datetime import date, datetime

from pydantic import BaseModel

from app.image.dto import PreviewTile
from app.image.entity.image import Image


class BoundsSchema(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


class FetchPreviewRequest(BaseModel):
    area_id: uuid.UUID
    date_start: date
    date_end: date


class PreviewTileResponse(BaseModel):
    image_id: uuid.UUID
    preview_url: str
    bounds: BoundsSchema


class SaveImagesRequest(BaseModel):
    area_id: uuid.UUID


class ImageResponse(BaseModel):
    id: uuid.UUID
    area_id: uuid.UUID
    source: str
    bounds: BoundsSchema
    created_at: datetime


def to_preview_response(tile: PreviewTile) -> PreviewTileResponse:
    return PreviewTileResponse(
        image_id=tile.image_id,
        preview_url=f"/images/{tile.image_id}/png",
        bounds=BoundsSchema(
            min_lat=tile.bounds.min_lat,
            min_lon=tile.bounds.min_lon,
            max_lat=tile.bounds.max_lat,
            max_lon=tile.bounds.max_lon,
        ),
    )


def to_image_response(image: Image) -> ImageResponse:
    return ImageResponse(
        id=image.id,
        area_id=image.area_id,
        source=image.source,
        bounds=BoundsSchema(
            min_lat=image.bounds.min_lat,
            min_lon=image.bounds.min_lon,
            max_lat=image.bounds.max_lat,
            max_lon=image.bounds.max_lon,
        ),
        created_at=image.created_at,
    )
