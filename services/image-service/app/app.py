from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from .database import DB_INITIALIZER, get_async_session
from . import config
from .database.models import Image


cfg: config.Config = config.load_config()

app = FastAPI(title='Image Service')


@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )