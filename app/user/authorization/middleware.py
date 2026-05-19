import re
from dataclasses import dataclass
from uuid import UUID

import jwt
from fastapi.responses import JSONResponse
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import load_config
from app.image.models.image import Image
from app.map.models.area import Area
from app.shared.database import get_session

_cfg = load_config()

_UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_PROTECTED_RE = re.compile(r"^/(areas|images|analysis|groups|user)")
_USER_IN_PATH_RE = re.compile(rf"/user/({_UUID_PATTERN})")
_AREA_IN_PATH_RE = re.compile(rf"/areas/({_UUID_PATTERN})")
_IMAGE_IN_PATH_RE = re.compile(rf"/images/({_UUID_PATTERN})/png$")


@dataclass
class Subject:
    user_id: UUID
    group_id: int
    owns_area: bool


@dataclass
class Resource:
    resource: str
    body: dict
    params: dict


class AuthorizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if not _PROTECTED_RE.match(path):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Not authenticated"})

        decoded = _decode_bearer(auth[7:])
        if decoded is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        
        body = {}
        params = {}
        user_id, group_id = decoded

        if m := _USER_IN_PATH_RE.search(path):
            params["user_id"] = UUID(m.group(1))

        request_body = await _get_request_body(request)
        if uid := request_body.get("user_id"):
            body["user_id"] = UUID(str(uid))

        if m := _AREA_IN_PATH_RE.search(path):
            params["area_id"] = UUID(m.group(1))
        elif m := _IMAGE_IN_PATH_RE.search(path):
            params["area_id"] = await _get_area_id_for_image(UUID(m.group(1)))
        elif body and (aid := body.get("area_id")):
            body["area_id"] = UUID(str(aid))
        elif qaid := request.query_params.get("area_id"):
            params["area_id"] = UUID(qaid)

        owns_area = False
        area_id = params.get("area_id") if params.get("area_id") else body.get("area_id")
        if area_id is not None:
            owns_area = await _check_area_ownership(area_id, user_id)

        sub = Subject(
            user_id=user_id,
            group_id=group_id,
            owns_area=owns_area,
        )
        obj = Resource(
            resource="/" + request.path_params["path_name"],
            body=body,
            params=params
        )

        enforcer = request.app.state.enforcer
        if not enforcer.enforce(sub, obj, request.method):
            return JSONResponse(status_code=403, content={"detail": "Forbidden"})

        return await call_next(request)


def _decode_bearer(token: str) -> tuple[UUID, int] | None:
    try:
        payload = jwt.decode(
            token,
            _cfg.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            audience=["fastapi-users:auth"],
        )
        return UUID(payload["sub"]), int(payload["group_id"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None


async def _get_request_body(request: Request) -> dict:
    request_body = {}
    try:
        request_body.update(await request.json())
    except Exception:
        pass
    return request_body


async def _get_area_id_for_image(image_id: UUID) -> UUID | None:
    async for session in get_session():
        result = await session.execute(
            select(Image.area_id).where(Image.id == image_id)
        )
        return result.scalar_one_or_none()
    return None


async def _check_area_ownership(area_id: UUID, user_id: UUID) -> bool:
    async for session in get_session():
        result = await session.execute(
            select(Area.user_id).where(Area.id == area_id)
        )
        owner_id = result.scalar_one_or_none()
        return owner_id == user_id
    return False
