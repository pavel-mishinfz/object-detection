from pydantic import BaseModel


class ObjectTypeBase(BaseModel):
    """
    Базовая модель типа объекта
    """
    name: str

    class ConfigDict:
        from_attribute = True


class ObjectType(ObjectTypeBase):
    """
    Модель для получения типа объекта
    """
    id: int


class ObjectTypeUpsert(ObjectType):
    """
    Модель для добавления/обновления типа объекта
    """
    pass
