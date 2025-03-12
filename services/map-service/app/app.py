from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from .database import DB_INITIALIZER, get_async_session
from .database.models import Area
from . import config


cfg: config.Config = config.load_config()

app = FastAPI(title='Map Service')


@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )