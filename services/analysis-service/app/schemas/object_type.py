from pydantic import BaseModel


class ObjectTypeUpsert(BaseModel):
    id: int
    name: str

    class ConfigDict:
        from_attribute = True