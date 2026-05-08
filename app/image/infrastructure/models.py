import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "image"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    area_id = Column(UUID(as_uuid=True), ForeignKey("area.id", ondelete="CASCADE"), nullable=False, index=True)
    source = Column(String(50), nullable=False)
    path = Column(String, nullable=False)
    bounds_min_lat = Column(Float, nullable=False)
    bounds_min_lon = Column(Float, nullable=False)
    bounds_max_lat = Column(Float, nullable=False)
    bounds_max_lon = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
