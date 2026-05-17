import uuid

from geoalchemy2 import Geometry
from sqlalchemy import Column, DateTime, String, UUID, UniqueConstraint
from sqlalchemy.sql import func

from app.shared.database import Base


class Area(Base):
    __tablename__ = "area"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_area_user_name"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    name = Column(String, nullable=False)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())
