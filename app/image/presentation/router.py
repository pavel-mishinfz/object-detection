import uuid

from fastapi import APIRouter, Depends, Response
from fastapi.responses import Response as FastAPIResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import load_config
from app.image.application import use_cases
from app.image.domain.errors import (
    AreaAccessDeniedError,
    AreaNotFoundError,
    ImageNotFoundError,
    InvalidDateRangeError,
    NoPreviewAvailableError,
)
from app.image.infrastructure.area_reader import AreaReader
from app.image.infrastructure.image_cache import RedisImageCache
from app.image.infrastructure.image_storage import FileImageStorage
from app.image.infrastructure.repository import ImageRepository
from app.image.infrastructure.sentinel_gateway import SentinelHubGateway
from app.image.presentation.schemas import (
    FetchPreviewRequest,
    ImageResponse,
    PreviewTileResponse,
    SaveImagesRequest,
    to_image_response,
    to_preview_response,
)

from fastapi import HTTPException

router = APIRouter(prefix="/images", tags=["images"])
cfg = load_config()


def _get_gateway() -> SentinelHubGateway:
    return SentinelHubGateway(
        client_id=cfg.sentinel_client_id,
        client_secret=cfg.sentinel_client_secret.get_secret_value(),
    )


def _get_cache() -> RedisImageCache:
    return RedisImageCache(
        host=cfg.redis_host,
        port=cfg.redis_port,
        db=cfg.redis_db,
    )


def _get_storage() -> FileImageStorage:
    return FileImageStorage(
        temp_dir=cfg.sentinel_temp_dir,
        images_dir=cfg.sentinel_images_dir,
    )


@router.post("/preview", response_model=list[PreviewTileResponse], status_code=200)
async def preview_images(
    payload: FetchPreviewRequest,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> list[PreviewTileResponse]:
    area_reader = AreaReader(session)
    cache = _get_cache()
    try:
        await use_cases.fetch_previews(
            area_id=payload.area_id,
            user_id=current_user_id,
            date_start=payload.date_start,
            date_end=payload.date_end,
            area_reader=area_reader,
            gateway=_get_gateway(),
            cache=cache,
            storage=_get_storage(),
        )
        tiles = await use_cases.get_preview_tiles(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_reader=area_reader,
            cache=cache,
        )
        return [to_preview_response(t) for t in tiles]
    except InvalidDateRangeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except AreaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AreaAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/save", response_model=list[ImageResponse], status_code=201)
async def save_images(
    payload: SaveImagesRequest,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> list[ImageResponse]:
    area_reader = AreaReader(session)
    repo = ImageRepository(session)
    try:
        await use_cases.save_images(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_reader=area_reader,
            repo=repo,
            storage=_get_storage(),
            cache=_get_cache(),
        )
        images = await use_cases.get_images(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_reader=area_reader,
            repo=repo,
        )
        return [to_image_response(img) for img in images]
    except NoPreviewAvailableError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except AreaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AreaAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("", response_model=list[ImageResponse])
async def get_images(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> list[ImageResponse]:
    try:
        images = await use_cases.get_images(
            area_id=area_id,
            user_id=current_user_id,
            area_reader=AreaReader(session),
            repo=ImageRepository(session),
        )
        return [to_image_response(img) for img in images]
    except AreaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AreaAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{image_id}/png")
async def get_image_png(
    image_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> FastAPIResponse:
    try:
        png_bytes = await use_cases.get_image_as_png(
            image_id=image_id,
            user_id=current_user_id,
            repo=ImageRepository(session),
            area_reader=AreaReader(session),
            storage=_get_storage(),
        )
        return FastAPIResponse(content=png_bytes, media_type="image/png")
    except ImageNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AreaAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("", status_code=204)
async def delete_images(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> Response:
    try:
        await use_cases.delete_images(
            area_id=area_id,
            user_id=current_user_id,
            area_reader=AreaReader(session),
            repo=ImageRepository(session),
            storage=_get_storage(),
        )
        return Response(status_code=204)
    except AreaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AreaAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
