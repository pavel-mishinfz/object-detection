import uuid
from pydantic import BaseModel


class DeviceBase(BaseModel):
    fingerprint: str
    user_id: uuid.UUID

    class ConfigDict:
        from_attribute = True

        
class DeviceIn(DeviceBase):
    pass


class Device(DeviceBase):
    id: uuid.UUID
