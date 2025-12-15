from fastapi_users.authentication import BearerTransport as BaseBearerTransport
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi_users.schemas import model_dump

from pydantic import BaseModel


class BearerResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class BearerTransport(BaseBearerTransport):
    async def get_login_response(self, access_token: str, refresh_token: str) -> Response:
        bearer_response = BearerResponse(
            access_token=access_token, 
            refresh_token=refresh_token, 
            token_type="bearer"
        )
        return JSONResponse(model_dump(bearer_response))
