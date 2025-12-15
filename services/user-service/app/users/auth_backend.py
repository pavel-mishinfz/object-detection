from typing import Callable

from fastapi import Response

from fastapi_users import models
from fastapi_users.authentication import AuthenticationBackend as BaseAuthenticationBackend
from fastapi_users.authentication.strategy import Strategy
from fastapi_users.authentication.transport import Transport
from fastapi_users.types import DependencyCallable


class AuthenticationBackend(BaseAuthenticationBackend):
    def __init__(
        self,
        name: str,
        transport: Transport,
        get_strategy: DependencyCallable[Strategy[models.UP, models.ID]],
        create_refresh_token: Callable
    ):
        self.name = name
        self.transport = transport
        self.get_strategy = get_strategy
        self.create_refresh_token = create_refresh_token

    async def login(
        self, 
        strategy: Strategy[models.UP, models.ID],
        user: models.UP,
    ) -> Response:
        access_token = await strategy.write_token(user)
        refresh_token = await self.create_refresh_token(user)
        return await self.transport.get_login_response(access_token=access_token, refresh_token=refresh_token)
