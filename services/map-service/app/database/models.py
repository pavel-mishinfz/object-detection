import uuid
from sqlalchemy import Column, DateTime, String, UUID
from sqlalchemy.sql import func
from geoalchemy2 import Geometry

from .database import Base


class Area(Base):
    __tablename__ = 'area'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    geometry = Column(Geometry(geometry_type='POLYGON', srid=4326), nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())



