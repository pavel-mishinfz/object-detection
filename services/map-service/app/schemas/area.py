import uuid
from typing import List, Tuple

from pydantic import BaseModel, Field, field_validator
from datetime import datetime

from shapely import wkb
from shapely.geometry import mapping


class GeoJSONPolygon(BaseModel):
    """
    Модель полигона в формате GeoJSON
    """
    type: str = Field("Polygon", title='Тип геометрии')
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
    geometry: GeoJSONPolygon = Field(title='Координаты вершин полигона в формате GeoJSON')
    name: str = Field(title='Наименование полигона')

    # class ConfigDict:
    #     from_attribute = True


class AreaIn(AreaBase):
    """
    Модель для добавления полигона в базу
    """
    user_id: uuid.UUID = Field(title='Идентификатор пользователя, создавшего полигон')


class AreaUpdate(AreaBase):
    """
    Модель для обновления полигона
    """
    pass


class Area(AreaBase):
    """
    Модель для получения полигона из базы
    """
    id: uuid.UUID = Field(title="Идентификатор полигона")
    user_id: uuid.UUID = Field(title='Идентификатор пользователя, создавшего полигон')
    created_at: datetime = Field(title="Время создания полигона")

    @field_validator("geometry", mode='before')
    @classmethod
    def convert_wkb_to_geojson(cls, value):
        """
        Преобразует WKBElement в GeoJSON
        """
        shape_geometry = wkb.loads(value.data)  # Преобразуем WKB в Shapely-объект
        return mapping(shape_geometry)  # Конвертируем в GeoJSON


class AreaSummary(BaseModel):
    """
    Возвращает основную информацию о полигоне (без геометрии)
    """
    id: uuid.UUID
    name: str
    created_at: datetime
