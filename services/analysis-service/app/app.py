import json

from fastapi import FastAPI, Depends, HTTPException

from .database import DB_INITIALIZER, get_async_session
from . import config
from . import crud
from . import schemas


cfg: config.Config = config.load_config()

app = FastAPI(title='Result Service')


@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )

    objects_type = []
    with open(cfg.default_objects_config_path, encoding="utf-8") as f:
        objects_type = json.load(f)

    async for session in get_async_session():
        for object_type in objects_type:
            await crud.upsert_object_type(
                session, schemas.object_type.ObjectTypeUpsert(**object_type)
            )