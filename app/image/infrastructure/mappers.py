from app.image.domain.image import Image, ImageBounds
from app.image.infrastructure.models import Image as ImageRecord


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
