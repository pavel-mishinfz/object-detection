import uuid
from sqlalchemy import Column, DateTime, String, UUID
from sqlalchemy.dialects.postgresql import JSONB

from .database import Base


class Image(Base):
    __tablename__ = 'image'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    area_id = Column(UUID(as_uuid=True), nullable=False)
    image_date = Column(DateTime)
    source = Column(String(50), nullable=False)
    url = Column(String, nullable=False)
    image_metadata = Column(JSONB)



