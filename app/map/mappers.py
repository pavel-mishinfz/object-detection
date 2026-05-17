from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Polygon as ShapelyPolygon

from app.map.models.area import Area
from app.map.entity.polygon import Coordinate, Polygon


def to_domain(record: Area) -> Polygon:
    return Polygon(
        id=record.id,
        user_id=record.user_id,
        name=record.name,
        coordinates=_wkb_to_coords(record.geometry),
        created_at=record.created_at,
    )


def to_orm(polygon: Polygon) -> Area:
    return Area(
        id=polygon.id,
        user_id=polygon.user_id,
        name=polygon.name,
        geometry=_coords_to_wkb(polygon.coordinates),
        created_at=polygon.created_at,
    )


def _coords_to_wkb(coords: tuple[Coordinate, ...]) -> WKBElement:
    shapely_poly = ShapelyPolygon([(c.lon, c.lat) for c in coords])
    return from_shape(shapely_poly, srid=4326)


def _wkb_to_coords(wkb) -> tuple[Coordinate, ...]:
    shapely_poly = to_shape(wkb)
    return tuple(
        Coordinate(lat=lat, lon=lon)
        for lat, lon in shapely_poly.exterior.coords
    )