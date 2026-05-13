from geoalchemy2.shape import to_shape

from app.analysis.domain.segmentation_result import ObjectType, SegmentationResult
from app.analysis.infrastructure.models import ObjectType as ObjectTypeRecord
from app.analysis.infrastructure.models import SegmentationRecord


def to_domain_object_type(record: ObjectTypeRecord) -> ObjectType:
    return ObjectType(id=record.id, name=record.name)


def to_domain_segmentation_result(record: SegmentationRecord) -> SegmentationResult:
    shapely_geom = to_shape(record.geometry)
    coords = tuple((lon, lat) for lon, lat in shapely_geom.exterior.coords)
    return SegmentationResult(
        id=record.id,
        area_id=record.area_id,
        image_id=record.image_id,
        geometry=coords,
        object_type=to_domain_object_type(record.object_type),
        created_at=record.created_at,
    )
