from typing import Callable

from fastapi import Response, status

from fastapi_users import models
from fastapi_users.authentication.strategy import (
    Strategy,
    StrategyDestroyNotSupportedError,
)
from fastapi_users.authentication.transport import (
    Transport,
    TransportLogoutNotSupportedError,
)
from fastapi_users.authentication import AuthenticationBackend as BaseAuthenticationBackend
from fastapi_users.types import DependencyCallable


class AuthenticationBackend(BaseAuthenticationBackend):
    def __init__(
        self,
        name: str,
        transport: Transport,
        get_strategy: DependencyCallable[Strategy[models.UP, models.ID]],
        get_init_strategy: Callable,
        create_refresh_token: Callable,
        delete_refresh_token: Callable,
    ):
        self.name = name
        self.transport = transport
        self.get_strategy = get_strategy
        self.get_init_strategy = get_init_strategy
        self.create_refresh_token = create_refresh_token
        self.delete_refresh_token = delete_refresh_token

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
    
    async def logout(
        self, strategy: Strategy[models.UP, models.ID], user: models.UP, token: str
    ) -> Response:
        
        await self.delete_refresh_token(user)

        try:
            await strategy.destroy_token(token, user)
        except StrategyDestroyNotSupportedError:
            pass

        try:
            response = await self.transport.get_logout_response()
        except TransportLogoutNotSupportedError:
            response = Response(status_code=status.HTTP_204_NO_CONTENT)

        return response

    async def refresh_tokens(
        self,
        user: models.UP,
    ) -> Response:
        strategy = await self.get_init_strategy()
        return await self.login(strategy, user)
