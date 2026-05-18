from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import Column, ForeignKey, String
from sqlalchemy.orm import mapped_column, relationship

from app.shared.database import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    username = Column(String(128), nullable=False)
    group_id = mapped_column(ForeignKey("group.id"), nullable=False)
    group = relationship("Group", uselist=False)
