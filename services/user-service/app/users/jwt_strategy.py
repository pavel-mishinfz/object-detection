from typing import Any

from fastapi_users.authentication import JWTStrategy
from fastapi_users.jwt import generate_jwt


class CustomJWTStrategy(JWTStrategy):
    async def write_token(self, user: Any) -> str:
        data = {"sub": str(user.id), "aud": self.token_audience, "group_id": user.group_id}
        return generate_jwt(
            data, self.encode_key, self.lifetime_seconds, algorithm=self.algorithm
        )
    