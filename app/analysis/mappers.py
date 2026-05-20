from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Polygon

from app.analysis.entity.segmentation_result import ObjectType, SegmentationResult
from app.analysis.models.segmentation_result import (
    ObjectType as ObjectTypeRecord, 
    SegmentationResult as SegmentationResultRecord
)


def to_domain_object_type(record: ObjectTypeRecord) -> ObjectType:
    return ObjectType(id=record.id, name=record.name)


def to_domain_segmentation_result(record: SegmentationResultRecord) -> SegmentationResult:
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


def to_orm_segmentation_result(result: SegmentationResult) -> SegmentationResultRecord:
    return SegmentationResultRecord(
        id=result.id,
        area_id=result.area_id,
        image_id=result.image_id,
        geometry=from_shape(Polygon(result.geometry), srid=4326),
        object_type_id=result.object_type.id,
        created_at=result.created_at,
    )


def to_orm_object_type(object_type: ObjectType) -> ObjectTypeRecord:
    return ObjectTypeRecord(id=object_type.id, name=object_type.name)
