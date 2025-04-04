import uuid

from fastapi import Depends, FastAPI
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import (AuthenticationBackend,
                                          BearerTransport,
                                          JWTStrategy)

from . import secretprovider, usermanager
from .database import models
from . import schemas
from typing import Any
from fastapi_users.jwt import generate_jwt


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


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
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