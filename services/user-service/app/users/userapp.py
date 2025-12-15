from datetime import datetime, timedelta
import uuid
from typing import Any

from fastapi import Depends, FastAPI
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTStrategy
from fastapi_users.jwt import generate_jwt

from . import secretprovider, usermanager, schemas, refreshcrud
from .auth_backend import AuthenticationBackend
from .bearer_transport import BearerTransport, BearerResponse
from .database import database, models


bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


class CustomJWTStrategy(JWTStrategy):
    async def write_token(self, user: Any) -> str:
        data = {"sub": str(user.id), "aud": self.token_audience, "group_id": user.group_id}
        return generate_jwt(
            data, self.encode_key, self.lifetime_seconds, algorithm=self.algorithm
        )


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
        refresh_token: models.RefreshToken = await refreshcrud.create_refresh_token(refresh_token_in, session)
    return str(refresh_token.refresh_token)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
    create_refresh_token=create_refresh_token
)

async def update_access_and_refresh_tokens(current_user: Any) -> BearerResponse:
    response: BearerResponse = await auth_backend.login(
        strategy=auth_backend.get_strategy,
        user=current_user
    )
    return response

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