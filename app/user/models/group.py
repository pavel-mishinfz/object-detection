from sqlalchemy import Column, ForeignKey, Integer, String

from app.shared.database import Base


class Group(Base):
    __tablename__ = "group"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String, nullable=False, unique=True)
