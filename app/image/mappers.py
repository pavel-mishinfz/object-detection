from app.image.entity.image import Image, ImageBounds
from app.image.models.image import Image as ImageRecord


def to_domain(record: ImageRecord) -> Image:
    return Image(
        id=record.id,
        area_id=record.area_id,
        source=record.source,
        path=record.path,
        bounds=ImageBounds(
            min_lat=record.bounds_min_lat,
            min_lon=record.bounds_min_lon,
            max_lat=record.bounds_max_lat,
            max_lon=record.bounds_max_lon,
        ),
        created_at=record.created_at,
    )


def to_orm(image: Image) -> ImageRecord:
    return ImageRecord(
        id=image.id,
        area_id=image.area_id,
        source=image.source,
        path=image.path,
        bounds_min_lat=image.bounds.min_lat,
        bounds_min_lon=image.bounds.min_lon,
        bounds_max_lat=image.bounds.max_lat,
        bounds_max_lon=image.bounds.max_lon,
        created_at=image.created_at,
    )
