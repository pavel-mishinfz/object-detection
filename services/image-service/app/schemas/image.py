import uuid
from typing import Optional, Dict

from pydantic import BaseModel, Field
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
    Модель для добавления снимка в БД
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
    geometry_wkb: bytes = Field(title='Координаты вершин полигона в формате WKB')
    date_start: date = Field(date.today() - timedelta(days=31), title='Дата начала периода съемки')
    date_end: date = Field(date.today(), title='Дата окончания периода съемки')
    resolution: int = Field(10, title='Разрешение снимка (в метрах)')
