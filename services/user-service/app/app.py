import json

from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Cookie, Body, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from . import config, users
from .users import schemas
from .users.database import database


app_config: config.Config = config.load_config()

app = FastAPI(title='User Service')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

users.include_routers(app)


@app.post(
    "/auth/jwt/refresh",
    summary='Обновляет access и refresh токены',
    tags=['auth']
    )
async def refresh_tokens(
    fingerprint: str = Body(..., embed=True),
    refresh_token: str = Cookie(),
    current_user=Depends(users.fastapi_users.current_user(active=True)),
    session: AsyncSession = Depends(database.get_async_session),
    ):
    refresh_token_info = await users.refreshcrud.get_refresh_token(session, refresh_token)
    if not refresh_token_info:
        raise HTTPException(
            status_code=400,
            detail='NOT_FOUND_REFRESH_SESSION'
        )
    await users.refreshcrud.delete_refresh_token(session, refresh_token)

    if datetime.now() > refresh_token_info.expires_in:
        raise HTTPException(
            status_code=400,
            detail='TOKEN_EXPIRED'
        )

    devices = await users.devicecrud.get_devices(session, current_user.id)
    fingerprints = [device.fingerprint for device in devices]
    if fingerprint not in fingerprints:
        raise HTTPException(
            status_code=400,
            detail='INVALID_REFRESH_SESSION'
        )

    return await users.auth_backend.refresh_tokens(current_user)


@app.post(
    "/users/device",
    response_model=schemas.device.Device,
    summary='Создает запись с fingerprint браузера пользователя',
    tags=['users']
    )
async def add_device(
    device: schemas.device.DeviceIn,
    session: AsyncSession = Depends(database.get_async_session)
    ):

    return await users.devicecrud.create_device(device, session)

@app.post(
    "/groups", status_code=201,
    response_model=schemas.group.GroupRead,
    summary='Создает новую группу пользователей',
    tags=['user-groups']
    )
async def add_group(
    group: schemas.group.GroupCreate,
    session: AsyncSession = Depends(database.get_async_session)
    ):

    return await users.groupcrud.create_group(group, session)


@app.get(
    "/groups",
    summary='Возвращает список групп пользователей',
    response_model=list[schemas.group.GroupRead],
    tags=['user-groups']
    )
async def get_group_list(
    session: AsyncSession = Depends(database.get_async_session),
    skip: int = 0,
    limit: int = 100
    ):

    return await users.groupcrud.get_groups(session, skip, limit)


@app.get("/groups/{group_id}", summary='Возвращает информацию о группе пользователей', tags=['user-groups'])
async def get_group_info(
    group_id: int, session: AsyncSession = Depends(database.get_async_session)
    ):

    group = await users.groupcrud.get_group(session, group_id)
    if group is not None:
        return group
    return HTTPException(status_code=404, detail="Группа не найдена")


@app.put("/groups/{group_id}", summary='Обновляет информацию о группе пользователей', tags=['user-groups'])
async def update_group(
    group_id: int,
    group: schemas.group.GroupUpdate,
    session: AsyncSession = Depends(database.get_async_session)
    ):

    group = await users.groupcrud.update_group(session, group_id, group)
    if group is not None:
        return group
    return HTTPException(status_code=404, detail="Группа не найдена")


@app.delete("/groups/{group_id}", summary='Удаляет информацию о группе пользователей', tags=['user-groups'])
async def delete_group(
    group_id: int,
    session: AsyncSession = Depends(database.get_async_session)
    ):

    group = await users.groupcrud.get_group(session, group_id)
    if await users.groupcrud.delete_group(session, group_id):
        return group
    return HTTPException(status_code=404, detail="Группа не найдена")


@app.on_event("startup")
async def on_startup():
    await database.DB_INITIALIZER.init_database(
        app_config.postgres_dsn_async.unicode_string()
    )

    groups = []
    with open(app_config.default_groups_config_path, encoding="utf-8") as f:
        groups = json.load(f)   

    async for session in database.get_async_session():
        for group in groups:
            await users.groupcrud.upsert_group(
                session, schemas.group.GroupUpsert(**group)
            )

