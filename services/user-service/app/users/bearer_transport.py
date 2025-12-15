from fastapi_users.authentication import BearerTransport as BaseBearerTransport
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi_users.schemas import model_dump

from pydantic import BaseModel


class BearerResponse(BaseModel):
    access_token: str
    token_type: str

class BearerTransport(BaseBearerTransport):
    async def get_login_response(self, access_token: str, refresh_token: str) -> Response:
        bearer_response = BearerResponse(
            access_token=access_token,
            token_type="bearer"
        )

        response = JSONResponse(model_dump(bearer_response))
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=30 * 24 * 60 * 60  # 30 дней в секундах
        )

        return response
