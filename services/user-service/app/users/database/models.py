import uuid

from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import Column, ForeignKey, Integer, String, UUID, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import mapped_column, relationship

from . import database


class RefreshToken(database.Base):
    __tablename__ = 'refresh_token'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    refresh_token = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    expires_in = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())
    user_id = mapped_column(ForeignKey("user.id"), nullable=False)
    user = relationship("User", back_populates="refresh_tokens")


class Group(database.Base):
    __tablename__ = 'group'

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String)


class DeviceFingerprint(database.Base):
    __tablename__ = 'device_fingerprint'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    fingerprint = Column(String, unique=True, nullable=False)
    user_id = mapped_column(ForeignKey("user.id"), nullable=False)
    user = relationship("User", back_populates="devices")


class User(SQLAlchemyBaseUserTableUUID, database.Base):
    username = Column(String(length=128), nullable=False)
    group_id = mapped_column(ForeignKey("group.id"), nullable=False)
    group = relationship("Group", uselist=False)
    devices = relationship("DeviceFingerprint", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
