import uuid
from uuid import UUID

from fastapi import Depends
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import AuthenticationBackend, BearerTransport

from app.user.infrastructure.jwt_strategy import get_jwt_strategy
from app.user.infrastructure.models import User
from app.user.infrastructure.user_manager import get_user_manager

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])


async def get_current_user_id(
    user: User = Depends(fastapi_users.current_user(active=True)),
) -> UUID:
    return user.id
