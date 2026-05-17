from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import mapped_column, relationship

from app.shared.database import Base


class Group(Base):
    __tablename__ = "group"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String, nullable=False, unique=True)


class User(SQLAlchemyBaseUserTableUUID, Base):
    username = Column(String(128), nullable=False)
    group_id = mapped_column(ForeignKey("group.id"), nullable=False)
    group = relationship("Group", uselist=False)
