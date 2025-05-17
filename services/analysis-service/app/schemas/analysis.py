import uuid
from typing import Tuple, List

from pydantic import BaseModel, Field, field_validator

from shapely import wkb
from shapely.geometry import mapping

from .object_type import ObjectType


class GeoJSONPolygon(BaseModel):
    """
    Модель полигона в формате GeoJSON для сегментации
    """
    type: str = Field("Polygon", title='Тип геометрии')
    coordinates: List[List[Tuple[float, float]]] = Field(
        default=[
            [
                [54.281558777056105, 44.706032726562505],
                [54.281558777056205, 44.706032726562605],
                [54.281458777056105, 44.706032726562505],
                [54.281558777056105, 44.706032726562505],
            ]
        ],
        title="Координаты полигона (внешний контур)"
    )


class AnalysisIn(BaseModel):
    """
    Модель для запроса анализа
    """
    polygon_id: uuid.UUID = Field(title='Идентификатор полигона')
    images_ids: List[uuid.UUID] = Field(title='Идентификаторы снимков')
    images_paths: List[str] = Field(title='Список путей к снимкам')


class Analysis(BaseModel):
    """
    Модель для получения информации об анализе
    """
    id: uuid.UUID = Field(title='Идентификатор результата анализа')
    polygon_id: uuid.UUID
    image_id: uuid.UUID = Field(title='Идентификатор снимка')
    geometry: GeoJSONPolygon = Field(title='Координаты полигона в формате GeoJSON')
    object_type: ObjectType = Field(title='Тип объекта')

    @field_validator("geometry", mode='before')
    @classmethod
    def convert_wkb_to_geojson(cls, value):
        """
        Преобразует WKBElement в GeoJSON
        """
        shape_geometry = wkb.loads(value.data)  # Преобразуем WKB в Shapely-объект
        return mapping(shape_geometry)  # Конвертируем в GeoJSON