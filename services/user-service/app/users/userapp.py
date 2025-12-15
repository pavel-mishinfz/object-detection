from datetime import datetime, timedelta
import uuid
from typing import Any

from fastapi import Depends, FastAPI
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTStrategy

from . import secretprovider, usermanager, schemas, refreshcrud
from .auth_backend import AuthenticationBackend
from .bearer_transport import BearerTransport
from .database import database, models
from .jwt_strategy import CustomJWTStrategy


bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy(
    secret_provider: secretprovider.SecretProvider = Depends(secretprovider.get_secret_provider)
    ) -> JWTStrategy:
    return CustomJWTStrategy(secret=secret_provider.jwt_secret, lifetime_seconds=3600)

async def create_refresh_token(user: Any) -> str:
    refresh_token_in = schemas.refresh_token.RefreshTokenIn(
        expires_in=datetime.now() + timedelta(days=30),
        user_id=user.id
    )
    async for session in database.get_async_session():
        current_refresh_token: models.RefreshToken | None = await refreshcrud.get_refresh_token_by_user_id(session, user.id)
        if current_refresh_token:
            await refreshcrud.delete_refresh_token(session, current_refresh_token.refresh_token.__str__())

        refresh_token: models.RefreshToken = await refreshcrud.create_refresh_token(refresh_token_in, session)
    return refresh_token.refresh_token.__str__()


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
    create_refresh_token=create_refresh_token,
    get_secret_provider=secretprovider.get_secret_provider
)

fastapi_users = FastAPIUsers[models.User, uuid.UUID](
    usermanager.get_user_manager,
    [auth_backend]
)


def include_routers(app: FastAPI):
    app.include_router(
        fastapi_users.get_auth_router(auth_backend),
        prefix="/auth/jwt",
        tags=["auth"]
    )
    app.include_router(
        fastapi_users.get_register_router(schemas.user.UserRead, schemas.user.UserCreate),
        prefix="/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_reset_password_router(),
        prefix="/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_verify_router(schemas.user.UserRead),
        prefix="/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_users_router(schemas.user.UserRead, schemas.user.UserUpdate),
        prefix="/users",
        tags=["users"],
    )