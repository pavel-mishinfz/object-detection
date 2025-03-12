from pydantic import BaseModel


class GroupCreate(BaseModel):
    name: str

    class ConfigDict:
        from_attribute = True


class GroupRead(BaseModel):
    id: int
    name: str

    class ConfigDict:
        from_attribute = True


class GroupUpsert(BaseModel):
    id: int
    name: str

    class ConfigDict:
        from_attribute = True


class GroupUpdate(BaseModel):
    name: str

    class ConfigDict:
        from_attribute = True