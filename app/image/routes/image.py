import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import Response as FastAPIResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.image.dependencies import (
    get_image_repository,
    get_image_storage,
    get_redis_cache,
    get_sentinel_gateway,
    get_area_reader,
    get_area_access_policy
)
from app.image.exceptions import InvalidDateRangeError, NoPreviewAvailableError
from app.image.infrastructure.image_storage import LocalImageStorage
from app.image.infrastructure.repository import ImageRepository
from app.image.infrastructure.image_cache import RedisImageCache
from app.image.infrastructure.sentinel_gateway import SentinelHubGateway
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
from app.shared.contracts import IAreaAccessPolicy, IEventPublisher
from app.shared.database import get_session
from app.shared.dependencies import get_event_publisher
from app.shared.exceptions import AccessDeniedError, NotFoundError
from app.user.dependencies import get_current_user_id

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/preview", response_model=list[PreviewTileResponse], status_code=200)
async def preview_images(
    payload: FetchPreviewRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    area_reader: IAreaReader = Depends(get_area_reader),
    sentinel_gateway: SentinelHubGateway = Depends(get_sentinel_gateway),
    local_storage: LocalImageStorage = Depends(get_image_storage),
    redis_cache: RedisImageCache = Depends(get_redis_cache)
) -> list[PreviewTileResponse]:
    try:
        await image_service.fetch_previews(
            area_id=payload.area_id,
            user_id=current_user_id,
            date_start=payload.date_start,
            date_end=payload.date_end,
            area_access_policy=area_access_policy,
            area_reader=area_reader,
            sentinel_gateway=sentinel_gateway,
            storage=local_storage,
            cache=redis_cache,
        )
        tiles = await image_service.get_preview_tiles(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
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
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: ImageRepository = Depends(get_image_repository),
    storage: LocalImageStorage = Depends(get_image_storage),
    redis_cache: RedisImageCache = Depends(get_redis_cache),
    session: AsyncSession = Depends(get_session),
) -> list[ImageResponse]:
    try:
        await image_service.save_images(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
            storage=storage,
            cache=redis_cache,
            session=session
        )
        images = await image_service.get_images(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
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
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: ImageRepository = Depends(get_image_repository),
) -> list[ImageResponse]:
    try:
        images = await image_service.get_images(
            area_id=area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
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
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: ImageRepository = Depends(get_image_repository),
    storage: LocalImageStorage = Depends(get_image_storage),
) -> FastAPIResponse:
    try:
        png_bytes = await image_service.get_image_as_png(
            image_id=image_id,
            user_id=current_user_id,
            repo=repo,
            area_access_policy=area_access_policy,
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
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: ImageRepository = Depends(get_image_repository),
    storage: LocalImageStorage = Depends(get_image_storage),
    publisher: IEventPublisher = Depends(get_event_publisher),
    session: AsyncSession = Depends(get_session),
) -> Response:
    try:
        await image_service.delete_images(
            area_id=area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
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
