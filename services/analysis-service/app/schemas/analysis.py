import uuid
from typing import Tuple, List

from pydantic import BaseModel, Field, field_validator

from shapely import wkb
from shapely.geometry import mapping

from .object_type import ObjectType


class GeoJSONBbox(BaseModel):
    """
    Модель ограничиващей рамки в формате GeoJSON
    """
    type: str = Field("Polygon", title='Тип геометрии')
    coordinates: List[List[Tuple[float, float]]] = Field([
      [
        [54.281558777056105, 44.706032726562505],
        [54.281558777056105, 44.706032726562505]
      ]
    ], title="Координаты ограничивающей рамки")


class AnalysisIn(BaseModel):
    """
    Модель для запроса анализа
    """
    polygon_id: uuid.UUID = Field(title='Идентификатор полигона')
    images_ids: List[uuid.UUID] = Field(title='Идентификаторы снимков')
    images_paths: List[str] = Field(title='Список путей к снимкам')


class Analysis(BaseModel):
    """
    Модель для получения иноформации об анализе
    """
    id: uuid.UUID = Field(title='Идентификатор результата анализа')
    polygon_id: uuid.UUID
    image_id: uuid.UUID = Field(title='Идентификатор снимка')
    geometry: GeoJSONBbox = Field(title='Координаты ограничивающих рамок в формате GeoJSON')
    score: float = Field(title='Уверенность предикта')
    object_type: ObjectType = Field(title='Тип объекта')

    @field_validator("geometry", mode='before')
    @classmethod
    def convert_wkb_to_geojson(cls, value):
        """
        Преобразует WKBElement в GeoJSON
        """
        shape_geometry = wkb.loads(value.data)  # Преобразуем WKB в Shapely-объект
        return mapping(shape_geometry)  # Конвертируем в GeoJSON
