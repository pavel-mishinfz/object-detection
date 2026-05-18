from typing import Any

from fastapi import Depends
from fastapi_users.authentication import JWTStrategy
from fastapi_users.jwt import generate_jwt

from app.user.infrastructure.secret_provider import SecretProvider, get_secret_provider

ACCESS_TOKEN_LIFETIME = 1800  # 30 минут


class CustomJWTStrategy(JWTStrategy):
    async def write_token(self, user: Any) -> str:
        data = {
            "sub": str(user.id),
            "aud": self.token_audience,
            "group_id": user.group_id,
        }
        return generate_jwt(
            data, self.encode_key, self.lifetime_seconds, algorithm=self.algorithm
        )


def get_jwt_strategy(
    secret_provider: SecretProvider = Depends(get_secret_provider),
) -> JWTStrategy:
    return CustomJWTStrategy(
        secret=secret_provider.jwt_secret,
        lifetime_seconds=ACCESS_TOKEN_LIFETIME,
    )
