import uuid
from datetime import datetime
from pydantic import BaseModel


class RefreshTokenBase(BaseModel):
    expires_in: datetime
    user_id: uuid.UUID

    class ConfigDict:
        from_attribute = True

        
class RefreshTokenIn(RefreshTokenBase):
    pass


class RefreshToken(RefreshTokenBase):
    id: uuid.UUID
    refresh_token: str
    created_at: datetime