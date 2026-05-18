from geoalchemy2.shape import to_shape, from_shape
from shapely.geometry import Polygon

from app.analysis.entity.detection_result import DetectionResult, ObjectType
from app.analysis.models.detection_result import DetectionResult as DetectionResultRecord
from app.analysis.models.detection_result import ObjectType as ObjectTypeRecord


def to_domain_object_type(record: ObjectTypeRecord) -> ObjectType:
    return ObjectType(id=record.id, name=record.name)


def to_domain_detection_result(record: DetectionResultRecord) -> DetectionResult:
    shapely_geom = to_shape(record.geometry)
    coords = tuple((lon, lat) for lon, lat in shapely_geom.exterior.coords)
    return DetectionResult(
        id=record.id,
        area_id=record.area_id,
        image_id=record.image_id,
        geometry=coords,
        score=float(record.score),
        object_type=to_domain_object_type(record.object_type),
        created_at=record.created_at,
    )


def to_orm_detection_result(detection_result: DetectionResult) -> DetectionResultRecord:
    return DetectionResultRecord(
        id=detection_result.id,
        area_id=detection_result.area_id,
        image_id=detection_result.image_id,
        geometry=from_shape(Polygon(detection_result.geometry), srid=4326),
        score=detection_result.score,
        object_type_id=detection_result.object_type.id,
        created_at=detection_result.created_at,
    )


def to_orm_object_type(object_type: ObjectType) -> ObjectTypeRecord:
    return ObjectTypeRecord(
        id=object_type.id, 
        name=object_type.name
    )