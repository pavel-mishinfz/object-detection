import uuid

from geoalchemy2 import Geometry
from sqlalchemy import Column, DateTime, Double, ForeignKey, Integer, String, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.shared.db import Base


class ObjectType(Base):
    __tablename__ = "object_type"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, unique=True)

    results = relationship("DetectionResult", back_populates="object_type")


class DetectionResult(Base):
    __tablename__ = "detection_result"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    area_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    image_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    score = Column(Double, nullable=False)
    object_type_id = Column(Integer, ForeignKey("object_type.id"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())

    object_type = relationship("ObjectType", back_populates="results", lazy="selectin")
