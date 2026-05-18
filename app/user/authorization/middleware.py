import re
from uuid import UUID

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import load_config
from app.map.models.area import Area
from app.shared.database import session_factory
from app.user.authorization.enforcer import AreaObject, UserSubject

_cfg = load_config()
_PROTECTED = re.compile(r'^/areas/([0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12})$')
_METHODS = frozenset({'GET', 'PUT', 'DELETE'})


class AreaOwnershipMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        m = _PROTECTED.match(request.url.path)
        if m and request.method in _METHODS:
            error = await _check(request, UUID(m.group(1)))
            if error:
                return error
        return await call_next(request)


async def _check(request: Request, polygon_id: UUID):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    try:
        payload = jwt.decode(
            auth[7:],
            _cfg.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            audience=["fastapi-users:auth"],
        )
        user_id = UUID(payload["sub"])
        group_id = int(payload["group_id"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return JSONResponse({"detail": "Invalid credentials"}, status_code=401)

    async with session_factory() as session:
        polygon = await session.get(Area, polygon_id)

    if polygon is None:
        return JSONResponse(
            {"detail": f"Полигон {polygon_id} не найден"}, status_code=404
        )

    enforcer = request.app.state.enforcer
    if not enforcer.enforce(
        UserSubject(id=user_id, group_id=group_id),
        AreaObject(owner_id=polygon.user_id, path=request.url.path),
        request.method,
    ):
        return JSONResponse(
            {"detail": "Нет доступа к данному полигону"}, status_code=403
        )

    request.state.current_user_id = user_id
    return None
