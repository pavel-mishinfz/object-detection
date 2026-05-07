from pyproj import Geod
from shapely.geometry import Polygon as ShapelyPolygon

from app.map.domain.polygon import Coordinate
from app.map.domain.errors import PolygonValidationError

_GEOD = Geod(ellps="WGS84")


def validate_closed(coords: tuple[Coordinate, ...]) -> None:
    if len(coords) < 2 or coords[0] != coords[-1]:
        raise PolygonValidationError("Полигон должен быть замкнут")


def validate_min_unique_coordinates(coords: tuple[Coordinate, ...]) -> None:
    unique = set(coords[:-1]) if coords and coords[0] == coords[-1] else set(coords)
    if len(unique) < 3:
        raise PolygonValidationError(
            "Полигон должен иметь не менее 3 уникальных координат"
        )


def validate_geographic_bounds(coords: tuple[Coordinate, ...]) -> None:
    for c in coords:
        if not (-90 <= c.lat <= 90):
            raise PolygonValidationError(
                f"Широта {c.lat} вне допустимого диапазона [-90, 90]"
            )
        if not (-180 <= c.lon <= 180):
            raise PolygonValidationError(
                f"Долгота {c.lon} вне допустимого диапазона [-180, 180]"
            )


def validate_no_self_intersections(coords: tuple[Coordinate, ...]) -> None:
    poly = ShapelyPolygon([(c.lon, c.lat) for c in coords])
    if not poly.is_valid:
        raise PolygonValidationError("Полигон содержит самопересечения")


def validate_area(coords: tuple[Coordinate, ...]) -> None:
    poly = ShapelyPolygon([(c.lon, c.lat) for c in coords])
    area_m2, _ = _GEOD.geometry_area_perimeter(poly)
    area_km2 = abs(area_m2) / 1_000_000
    if area_km2 < 1:
        raise PolygonValidationError(
            f"Площадь {area_km2:.4f} кв.км меньше минимальной (1 кв.км)"
        )
    if area_km2 > 2000:
        raise PolygonValidationError(
            f"Площадь {area_km2:.2f} кв.км превышает максимальную (2000 кв.км)"
        )


def validate_name(name: str) -> None:
    if not name or not name.strip():
        raise PolygonValidationError("Имя полигона не может быть пустым")


def validate_polygon_geometry(coords: tuple[Coordinate, ...]) -> None:
    validate_min_unique_coordinates(coords)
    validate_closed(coords)
    validate_geographic_bounds(coords)
    validate_no_self_intersections(coords)
    validate_area(coords)
