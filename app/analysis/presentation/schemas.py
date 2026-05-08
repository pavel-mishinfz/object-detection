import uuid
from datetime import datetime

from pydantic import BaseModel

from app.analysis.domain.detection_result import DetectionResult, ObjectType


class ObjectTypeResponse(BaseModel):
    id: int
    name: str


class GeoJSONPolygon(BaseModel):
    type: str = "Polygon"
    coordinates: list[list[tuple[float, float]]]


class DetectionResultResponse(BaseModel):
    id: uuid.UUID
    area_id: uuid.UUID
    image_id: uuid.UUID
    geometry: GeoJSONPolygon
    score: float
    object_type: ObjectTypeResponse
    created_at: datetime


class RunAnalysisRequest(BaseModel):
    area_id: uuid.UUID


def to_object_type_response(ot: ObjectType) -> ObjectTypeResponse:
    return ObjectTypeResponse(id=ot.id, name=ot.name)


def to_detection_result_response(result: DetectionResult) -> DetectionResultResponse:
    return DetectionResultResponse(
        id=result.id,
        area_id=result.area_id,
        image_id=result.image_id,
        geometry=GeoJSONPolygon(coordinates=[list(result.geometry)]),
        score=result.score,
        object_type=to_object_type_response(result.object_type),
        created_at=result.created_at,
    )
