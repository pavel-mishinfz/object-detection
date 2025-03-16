import uuid
from typing import List, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime

from geoalchemy2.shape import to_shape
from shapely.geometry import mapping

class GeoJSONPolygon(BaseModel):
    """
    Модель полигона в формате GeoJSON
    """
    type: str = Field("Polygon")
    coordinates: List[List[Tuple[float, float]]] = Field([
      [
        [54.281558777056105, 44.706032726562505],
        [54.37306375398213, 45.27457520703124],
        [54.06401982044115, 45.68656251171873],
        [54.281558777056105, 44.706032726562505]
      ]
    ], title="Координаты вершин полигона")

    @field_validator('coordinates')
    @classmethod
    def check_polygon(cls, value):
        # Проверка, что полигон замкнут
        if value[0][0] != value[0][-1]:
            raise ValueError("Polygon must be closed!")
        return value


class AreaBase(BaseModel):
    """
    Базовая модель полигона
    """
    user_id: uuid.UUID = Field(title='Идентификатор пользователя, создавшего полигон')
    geometry: GeoJSONPolygon = Field(title='Координаты вершин полигона в формате GeoJSON')
    name: str = Field(title='Наименование полигона')

    class ConfigDict:
        from_attribute = True


class AreaIn(AreaBase):
    """
    Модель для добавления полигона в базу
    """
    pass


class Area(AreaBase):
    """
    Модель для получения полигона из базы
    """
    id: uuid.UUID = Field(title="Идентификатор полигона")
    created_at: datetime = Field(title="Время создания полигона")
