import uuid

from sqlalchemy import Column, DateTime, Float, String, UUID
from sqlalchemy.sql import func

from app.shared.db import Base


class Image(Base):
    __tablename__ = "image"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    area_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    source = Column(String(50), nullable=False)
    path = Column(String, nullable=False)
    bounds_min_lat = Column(Float, nullable=False)
    bounds_min_lon = Column(Float, nullable=False)
    bounds_max_lat = Column(Float, nullable=False)
    bounds_max_lon = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
