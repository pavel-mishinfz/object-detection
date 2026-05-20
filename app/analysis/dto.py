from dataclasses import dataclass


@dataclass(frozen=True)
class RawContour:
    geo_polygon: tuple[tuple[float, float], ...]  # (lon, lat), closed ring
    object_type_id: int
