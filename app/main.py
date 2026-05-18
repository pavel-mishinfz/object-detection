import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.analysis.entity.detection_result import ObjectType
from app.analysis.infrastructure.repository import ObjectTypeRepository
from app.analysis.routes.analysis import router as analysis_router
from app.analysis.services.event_handlers import (
    on_area_deleted as analysis_on_area_deleted,
    on_images_deleted as analysis_on_images_deleted,
)
from app.config import load_config
from app.image.infrastructure.image_storage import LocalImageStorage
from app.image.services.event_handlers import make_on_area_deleted as make_image_on_area_deleted
from app.image.routes.image import router as image_router
from app.shared.database import get_session, init_db
from app.shared.event_bus import EventBus
from app.shared.events import AreaDeleted, ImagesByAreaDeleted
from app.map.routes.area import router as map_router
from app.user.application import use_cases as user_use_cases
from app.user.infrastructure.group_repository import GroupRepository
from app.user.presentation.router import router as user_router

cfg = load_config()


async def _seed_groups() -> None:
    with open(cfg.default_groups_config_path, encoding="utf-8") as f:
        groups = json.load(f)
    async for session in get_session():
        repo = GroupRepository(session)
        for g in groups:
            await user_use_cases.upsert_group(group_id=g["id"], name=g["name"], repo=repo)
        await session.commit()


async def _seed_object_types() -> None:
    with open(cfg.default_objects_config_path, encoding="utf-8") as f:
        objects = json.load(f)
    async for session in get_session():
        repo = ObjectTypeRepository(session)
        for o in objects:
            await repo.upsert(ObjectType(id=o["id"], name=o["name"]))
        await session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    bus = EventBus()
    app.state.event_bus = bus
    await init_db()
    await _seed_groups()
    await _seed_object_types()
    bus.subscribe(AreaDeleted, analysis_on_area_deleted)
    bus.subscribe(ImagesByAreaDeleted, analysis_on_images_deleted)
    image_storage = LocalImageStorage(cfg.sentinel_temp_dir, cfg.sentinel_images_dir)
    bus.subscribe(AreaDeleted, make_image_on_area_deleted(image_storage))
    yield


application = FastAPI(title="Object Detection", lifespan=lifespan)

application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

application.include_router(user_router)
application.include_router(map_router)
application.include_router(image_router)
application.include_router(analysis_router)
