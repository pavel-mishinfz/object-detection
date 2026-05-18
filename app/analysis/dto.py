from dataclasses import dataclass


@dataclass(frozen=True)
class RawDetection:
    geo_polygon: tuple[tuple[float, float], ...]  # (lon, lat), closed ring
    score: float
    object_type_id: int
