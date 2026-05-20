import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import Response as FastAPIResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.image.dependencies import (
    get_image_repository,
    get_image_storage,
    get_redis_cache,
    get_sentinel_gateway,
    get_area_reader
)
from app.image.exceptions import InvalidDateRangeError, NoPreviewAvailableError
from app.image.interfaces.image_cache import IImageCache
from app.image.interfaces.image_storage import IImageStorage
from app.image.interfaces.repository import IImageRepository
from app.image.interfaces.sentinel_gateway import ISentinelGateway
from app.image.schemas.image import (
    FetchPreviewRequest,
    ImageResponse,
    PreviewTileResponse,
    SaveImagesRequest,
    to_image_response,
    to_preview_response,
)
from app.image.services import image_service
from app.image.contracts import IAreaReader
from app.shared.contracts import IEventPublisher
from app.shared.database import get_session
from app.shared.dependencies import get_event_publisher
from app.shared.exceptions import AccessDeniedError, NotFoundError

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/preview", response_model=list[PreviewTileResponse], status_code=200)
async def preview_images(
    payload: FetchPreviewRequest,
    area_reader: IAreaReader = Depends(get_area_reader),
    sentinel_gateway: ISentinelGateway = Depends(get_sentinel_gateway),
    local_storage: IImageStorage = Depends(get_image_storage),
    redis_cache: IImageCache = Depends(get_redis_cache)
) -> list[PreviewTileResponse]:
    try:
        await image_service.fetch_previews(
            area_id=payload.area_id,
            date_start=payload.date_start,
            date_end=payload.date_end,
            area_reader=area_reader,
            sentinel_gateway=sentinel_gateway,
            storage=local_storage,
            cache=redis_cache,
        )
        tiles = await image_service.get_preview_tiles(
            area_id=payload.area_id,
            cache=redis_cache,
        )
    except InvalidDateRangeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return [to_preview_response(t) for t in tiles]


@router.post("/save", response_model=list[ImageResponse], status_code=201)
async def save_images(
    payload: SaveImagesRequest,
    repo: IImageRepository = Depends(get_image_repository),
    storage: IImageStorage = Depends(get_image_storage),
    redis_cache: IImageCache = Depends(get_redis_cache),
    session: AsyncSession = Depends(get_session),
) -> list[ImageResponse]:
    try:
        await image_service.save_images(
            area_id=payload.area_id,
            repo=repo,
            storage=storage,
            cache=redis_cache,
            session=session
        )
        images = await image_service.get_images(
            area_id=payload.area_id,
            repo=repo,
        )
    except NoPreviewAvailableError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return [to_image_response(img) for img in images]


@router.get("", response_model=list[ImageResponse])
async def get_images(
    area_id: uuid.UUID,
    repo: IImageRepository = Depends(get_image_repository),
) -> list[ImageResponse]:
    try:
        images = await image_service.get_images(
            area_id=area_id,
            repo=repo,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return [to_image_response(img) for img in images]


@router.get("/{image_id}/png")
async def get_image_as_png(
    image_id: uuid.UUID,
    repo: IImageRepository = Depends(get_image_repository),
    storage: IImageStorage = Depends(get_image_storage),
) -> FastAPIResponse:
    try:
        png_bytes = await image_service.get_image_as_png(
            image_id=image_id,
            repo=repo,
            storage=storage,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return FastAPIResponse(content=png_bytes, media_type="image/png")


@router.delete("", status_code=204)
async def delete_images(
    area_id: uuid.UUID,
    repo: IImageRepository = Depends(get_image_repository),
    storage: IImageStorage = Depends(get_image_storage),
    publisher: IEventPublisher = Depends(get_event_publisher),
    session: AsyncSession = Depends(get_session),
) -> Response:
    try:
        await image_service.delete_images(
            area_id=area_id,
            repo=repo,
            storage=storage,
            publisher=publisher,
            session=session
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return Response(status_code=204)
