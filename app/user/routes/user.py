from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.database import get_session
from app.user.dependencies import get_group_repository
from app.user.exceptions import GroupNotFoundError, GroupValidationError
from app.user.infrastructure.auth_backend import auth_backend, fastapi_users
from app.user.infrastructure.group_repository import GroupRepository
from app.user.schemas.group import (
    CreateGroupRequest,
    UpdateGroupReqeust,
    GroupResponse,
    to_group_response
)
from app.user.schemas.user import (
    UserCreate,
    UserRead,
    UserUpdate,
)
from app.user.services import group_service

router = APIRouter()

# --- fastapi-users sub-routers ---

router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

# --- Groups CRUD ---

groups_router = APIRouter(prefix="/groups", tags=["groups"])


@groups_router.post("/", response_model=GroupResponse, status_code=201)
async def create_group(
    payload: CreateGroupRequest,
    repo: GroupRepository = Depends(get_group_repository),
    session: AsyncSession = Depends(get_session),
) -> GroupResponse:
    try:
        await group_service.create_group(name=payload.name, repo=repo, session=session)
        group = await group_service.get_group_by_name(name=payload.name, repo=repo)
    except GroupValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except GroupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return to_group_response(group)


@groups_router.get("/", response_model=list[GroupResponse])
async def get_groups(
    skip: int = 0,
    limit: int = 100,
    repo: GroupRepository = Depends(get_group_repository),
) -> list[GroupResponse]:
    groups = await group_service.get_groups(skip=skip, limit=limit, repo=repo)
    return [to_group_response(g) for g in groups]


@groups_router.get("/{group_id}", response_model=GroupResponse)
async def get_group(
    group_id: int,
    repo: GroupRepository = Depends(get_group_repository),
) -> GroupResponse:
    try:
        group = await group_service.get_group(group_id=group_id, repo=repo)
    except GroupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return to_group_response(group)


@groups_router.put("/{group_id}", response_model=GroupResponse)
async def update_group(
    group_id: int,
    payload: UpdateGroupReqeust,
    repo: GroupRepository = Depends(get_group_repository),
    session: AsyncSession = Depends(get_session),
) -> GroupResponse:
    try:
        await group_service.update_group(
            group_id=group_id, name=payload.name, repo=repo, session=session
        )
    except GroupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GroupValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    group = await group_service.get_group(group_id=group_id, repo=repo)
    return to_group_response(group)


@groups_router.delete("/{group_id}", status_code=204)
async def delete_group(
    group_id: int,
    repo: GroupRepository = Depends(get_group_repository),
    session: AsyncSession = Depends(get_session),
) -> Response:
    try:
        await group_service.delete_group(group_id=group_id, repo=repo, session=session)
    except GroupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(status_code=204)


router.include_router(groups_router)
