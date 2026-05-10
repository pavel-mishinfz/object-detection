from fastapi import APIRouter, Depends, HTTPException, Response

from app.composition import get_group_repository
from app.user.application import use_cases
from app.user.application.interfaces import IGroupRepository
from app.user.domain.errors import GroupNotFoundError, GroupValidationError
from app.user.infrastructure.auth_backend import auth_backend, fastapi_users
from app.user.presentation.schemas import (
    GroupCreate,
    GroupRead,
    GroupUpdate,
    UserCreate,
    UserRead,
    UserUpdate,
    to_group_response,
)

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


@groups_router.post("/", response_model=GroupRead, status_code=201)
async def create_group(
    payload: GroupCreate,
    repo: IGroupRepository = Depends(get_group_repository),
) -> GroupRead:
    try:
        await use_cases.create_group(name=payload.name, repo=repo)
    except GroupValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    group = await use_cases.get_group_by_name(name=payload.name, repo=repo)
    return to_group_response(group)


@groups_router.get("/", response_model=list[GroupRead])
async def get_groups(
    skip: int = 0,
    limit: int = 100,
    repo: IGroupRepository = Depends(get_group_repository),
) -> list[GroupRead]:
    groups = await use_cases.get_groups(skip=skip, limit=limit, repo=repo)
    return [to_group_response(g) for g in groups]


@groups_router.get("/{group_id}", response_model=GroupRead)
async def get_group(
    group_id: int,
    repo: IGroupRepository = Depends(get_group_repository),
) -> GroupRead:
    try:
        group = await use_cases.get_group(group_id=group_id, repo=repo)
    except GroupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return to_group_response(group)


@groups_router.put("/{group_id}", response_model=GroupRead)
async def update_group(
    group_id: int,
    payload: GroupUpdate,
    repo: IGroupRepository = Depends(get_group_repository),
) -> GroupRead:
    try:
        await use_cases.update_group(group_id=group_id, name=payload.name, repo=repo)
    except GroupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except GroupValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    group = await use_cases.get_group(group_id=group_id, repo=repo)
    return to_group_response(group)


@groups_router.delete("/{group_id}", status_code=204)
async def delete_group(
    group_id: int,
    repo: IGroupRepository = Depends(get_group_repository),
) -> Response:
    try:
        await use_cases.delete_group(group_id=group_id, repo=repo)
    except GroupNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(status_code=204)


router.include_router(groups_router)
