from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Polygon as ShapelyPolygon

from app.map.domain.polygon import Coordinate, Polygon
from app.map.infrastructure.models import Area


def to_domain(record: Area) -> Polygon:
    return Polygon(
        id=record.id,
        user_id=record.user_id,
        name=record.name,
        coordinates=wkb_to_coords(record.geometry),
        created_at=record.created_at,
    )


def coords_to_wkb(coords: tuple[Coordinate, ...]):
    shapely_poly = ShapelyPolygon([(c.lon, c.lat) for c in coords])
    return from_shape(shapely_poly, srid=4326)


def wkb_to_coords(wkb) -> tuple[Coordinate, ...]:
    shapely_poly = to_shape(wkb)
    return tuple(
        Coordinate(lat=lat, lon=lon)
        for lat, lon in shapely_poly.exterior.coords
    )
