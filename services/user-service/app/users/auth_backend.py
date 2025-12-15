from typing import Callable

from fastapi import Response

from fastapi_users import models
from fastapi_users.authentication import AuthenticationBackend as BaseAuthenticationBackend
from fastapi_users.authentication.strategy import Strategy
from fastapi_users.authentication.transport import Transport
from fastapi_users.types import DependencyCallable

from .jwt_strategy import CustomJWTStrategy


class AuthenticationBackend(BaseAuthenticationBackend):
    def __init__(
        self,
        name: str,
        transport: Transport,
        get_strategy: DependencyCallable[Strategy[models.UP, models.ID]],
        create_refresh_token: Callable,
        get_secret_provider: Callable
    ):
        self.name = name
        self.transport = transport
        self.get_strategy = get_strategy
        self.create_refresh_token = create_refresh_token
        self.get_secret_provider = get_secret_provider

    async def login(
        self, 
        strategy: Strategy[models.UP, models.ID],
        user: models.UP,
    ) -> Response:
        access_token = await strategy.write_token(user)
        refresh_token = await self.create_refresh_token(user)
        return await self.transport.get_login_response(
            access_token=access_token, refresh_token=refresh_token
        )

    async def refresh_tokens(
        self,
        user: models.UP,
    ) -> Response:
        secret_provider_gen = self.get_secret_provider()
        secret_provider = None
        async for provider in secret_provider_gen:
            secret_provider = provider
            break
        
        jwt_secret = secret_provider.jwt_secret
        strategy = CustomJWTStrategy(secret=jwt_secret, lifetime_seconds=3600*24*30)
        return await self.login(strategy, user)
