import uuid
import json
import hashlib
from typing import Optional, Dict

from pydantic import BaseModel, Field, validator
from datetime import date, datetime, timedelta


class ImageBase(BaseModel):
    """
    Базовая модель снимка
    """
    area_id: uuid.UUID = Field(title='Идентификатор полигона')
    image_date: Optional[datetime] = Field(None, title='Дата создания снимка')
    source: str = Field(title='Источник снимка (наименование спутника)')
    url: str = Field(title='Путь к снимку в хранилище')
    image_metadata: Optional[Dict] = Field(None, title='Мета-информация о снимке')


class ImageIn(ImageBase):
    """
    Модель для обавления снимка в БД
    """
    pass


class Image(ImageBase):
    """
    Модель для получения информации о снимке
    """
    id: uuid.UUID = Field(title='Идентификатор снимка')


class PolygonMeta(BaseModel):
    """
    Модель мета-информации о полигоне для запроса снимков
    """
    id: uuid.UUID = Field(title='Идентификатор полигона')
    geometry_geojson: dict = Field({
        "type": "Polygon",
        "coordinates": [
            [
                [54.281558777056105, 44.706032726562505],
                [54.37306375398213, 45.27457520703124],
                [54.06401982044115, 45.68656251171873],
                [54.281558777056105, 44.706032726562505]
            ]
        ]
    }, title='Координаты вершин полигона в формате GeoJSON')
    date_start: date = Field(date.today() - timedelta(days=31), title='Дата начала периода съемки')
    date_end: date = Field(date.today(), title='Дата окончания периода съемки')
    resolution: int = Field(10, title='Разрешение снимка (в метрах)')
    hash: str = Field("", title='Хэш полигона для уникальной идентификации')

    @validator('hash', always=True)
    def compute_hash(cls, v, values):
        """
        Вычисляет хэш на основе координат полигона и дат
        """
        # Нормализация координат (сортировка для устранения зависимости от порядка точек)
        coords = values['geometry_geojson']['coordinates'][0]
        normalized_coords = sorted(coords)

        # Подготовка данных для хэширования
        data = {
            "coordinates": normalized_coords,
            "date_start": values['date_start'].isoformat(),
            "date_end": values['date_end'].isoformat(),
        }
        data_str = json.dumps(data, sort_keys=True)  # sort_keys для стабильности

        # Вычисление SHA-256 хэша
        hash_hex = hashlib.sha256(data_str.encode()).hexdigest()

        return hash_hex

