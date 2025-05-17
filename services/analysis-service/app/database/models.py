import uuid
from sqlalchemy import Column, ForeignKey, Integer, String, UUID
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

from .database import Base


class SegmentationResult(Base):
    __tablename__ = 'segmentation_result'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    polygon_id = Column(UUID(as_uuid=True), nullable=False)
    image_id = Column(UUID(as_uuid=True), nullable=False)
    geometry = Column(Geometry(geometry_type='GEOMETRY', srid=4326), nullable=False)
    object_type_id = Column(Integer, ForeignKey('object_type.id'), nullable=False)

    object_type = relationship('ObjectType', back_populates='results', lazy='selectin')


class ObjectType(Base):
    __tablename__ = 'object_type'

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String(50), nullable=False, unique=True)

    results = relationship('SegmentationResult', back_populates='object_type')
