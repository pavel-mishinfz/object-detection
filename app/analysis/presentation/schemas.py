import uuid
from datetime import datetime

from pydantic import BaseModel

from app.analysis.domain.segmentation_result import ObjectType, SegmentationResult


class ObjectTypeResponse(BaseModel):
    id: int
    name: str


class GeoJSONPolygon(BaseModel):
    type: str = "Polygon"
    coordinates: list[list[tuple[float, float]]]


class SegmentationResultResponse(BaseModel):
    id: uuid.UUID
    area_id: uuid.UUID
    image_id: uuid.UUID
    geometry: GeoJSONPolygon
    object_type: ObjectTypeResponse
    created_at: datetime


class RunSegmentationRequest(BaseModel):
    area_id: uuid.UUID
    model_name: str


def to_object_type_response(ot: ObjectType) -> ObjectTypeResponse:
    return ObjectTypeResponse(id=ot.id, name=ot.name)


def to_segmentation_result_response(result: SegmentationResult) -> SegmentationResultResponse:
    return SegmentationResultResponse(
        id=result.id,
        area_id=result.area_id,
        image_id=result.image_id,
        geometry=GeoJSONPolygon(coordinates=[list(result.geometry)]),
        object_type=to_object_type_response(result.object_type),
        created_at=result.created_at,
    )
