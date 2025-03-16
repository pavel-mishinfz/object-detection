from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from .database import DB_INITIALIZER, get_async_session

from .schemas import Area, AreaIn

from . import crud
from . import config


cfg: config.Config = config.load_config()

app = FastAPI(title='Map Service')

@app.post('/areas', response_model=Area, summary='Добавляет полигон в базу', tags=["areas"])
async def add_area(
        area_in: AreaIn,
        db: AsyncSession = Depends(get_async_session)
):
    return await crud.create_area(db, area_in)

@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )