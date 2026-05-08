from geoalchemy2.shape import to_shape

from app.analysis.domain.detection_result import DetectionResult, ObjectType
from app.analysis.infrastructure.models import DetectionResult as DetectionResultRecord
from app.analysis.infrastructure.models import ObjectType as ObjectTypeRecord


def to_domain_object_type(record: ObjectTypeRecord) -> ObjectType:
    return ObjectType(id=record.id, name=record.name)


def to_domain_detection_result(record: DetectionResultRecord) -> DetectionResult:
    shapely_geom = to_shape(record.geometry)
    coords = tuple((lon, lat) for lon, lat in shapely_geom.coords)
    return DetectionResult(
        id=record.id,
        area_id=record.area_id,
        image_id=record.image_id,
        geometry=coords,
        score=float(record.score),
        object_type=to_domain_object_type(record.object_type),
        created_at=record.created_at,
    )
