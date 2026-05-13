import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.analysis.application.event_handlers import (
    on_area_deleted as analysis_on_area_deleted,
    on_images_deleted as analysis_on_images_deleted
)
from app.analysis.domain.segmentation_result import ObjectType
from app.analysis.infrastructure.segmentation_engine import UNetSegmentationEngine
from app.analysis.infrastructure.repository import ObjectTypeRepository
from app.analysis.presentation.router import router as analysis_router
from app.composition import set_segmentation_engine
from app.config import load_config
from app.image.application.event_handlers import on_area_deleted as image_on_area_deleted
from app.image.presentation.router import router as image_router
from app.map.presentation.router import router as map_router
from app.shared import event_bus
from app.shared.db import get_session, init_db
from app.shared.events import AreaDeleted, ImagesDeleted
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
async def lifespan(_: FastAPI):
    await init_db()
    await _seed_groups()
    await _seed_object_types()
    set_segmentation_engine(UNetSegmentationEngine(cfg.model_dir))
    event_bus.subscribe(AreaDeleted, analysis_on_area_deleted)
    event_bus.subscribe(ImagesDeleted, analysis_on_images_deleted)
    event_bus.subscribe(AreaDeleted, image_on_area_deleted)
    yield


application = FastAPI(title="Semantic Segmentation", lifespan=lifespan)

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
