import uuid

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from .database import DB_INITIALIZER, get_async_session

from .schemas import Area, AreaSummary, AreaIn, AreaUpdate

from . import crud
from . import config


cfg: config.Config = config.load_config()

app = FastAPI(title='Map Service')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    '/areas',
    response_model=Area,
    summary='Добавляет полигон в базу',
    tags=["areas"]
)
async def add_area(
        area_in: AreaIn,
        db: AsyncSession = Depends(get_async_session)
):
    return await crud.create_area(db, area_in)


@app.get(
    '/areas/{area_id}',
    response_model=Area,
    summary='Возвращает информацию о полигоне',
    tags=['areas']
)
async def get_area(
        area_id: uuid.UUID,
        db: AsyncSession = Depends(get_async_session)
):
    area = await crud.get_area(db, area_id)
    if area is None:
        raise HTTPException(status_code=404, detail="Полигон не найден")
    return area


@app.get(
    '/areas',
    response_model=list[AreaSummary],
    summary='Возвращает список полигонов пользователя',
    tags=['areas']
)
async def get_list_areas(
        user_id: uuid.UUID,
        db: AsyncSession = Depends(get_async_session)
):
    return await crud.get_list_areas(db, user_id)


@app.put(
    '/areas/{area_id}',
    response_model=Area,
    summary='Обновляет полигон',
    tags=['areas']
)
async def update_area(
        area_id: uuid.UUID,
        area_update: AreaUpdate,
        db: AsyncSession = Depends(get_async_session)
):
    area = await crud.update_area(db, area_id, area_update)
    if area is None:
        raise HTTPException(status_code=404, detail="Полигон не найден")
    return area


@app.delete(
    '/areas/{area_id}',
    response_model=Area,
    summary='Удаляет полигон',
    tags=['areas']
)
async def delete_area(
        area_id: uuid.UUID,
        db: AsyncSession = Depends(get_async_session)
):
    area = await crud.delete_area(db, area_id)
    if area is None:
        raise HTTPException(status_code=404, detail="Полигон не найден")
    return area


@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )