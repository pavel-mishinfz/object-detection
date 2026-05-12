import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import Response as FastAPIResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.composition import (
    get_area_access_policy,
    get_area_reader,
    get_current_user_id,
    get_image_cache,
    get_image_repository,
    get_image_storage,
    get_sentinel_gateway,
    get_event_publisher
)
from app.image.application import use_cases
from app.image.application.interfaces import (
    IAreaAccessPolicy,
    IAreaReader,
    IImageCache,
    IImageRepository,
    IImageStorage,
    ISentinelGateway,
    IEventPublisher
)
from app.image.domain.errors import (
    ImageNotFoundError,
    InvalidDateRangeError,
    NoPreviewAvailableError,
)
from app.image.presentation.schemas import (
    FetchPreviewRequest,
    ImageResponse,
    PreviewTileResponse,
    SaveImagesRequest,
    to_image_response,
    to_preview_response,
)
from app.shared.db import get_session
from app.shared.errors import AccessDeniedError, NotFoundError

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/preview", response_model=list[PreviewTileResponse], status_code=200)
async def preview_images(
    payload: FetchPreviewRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    area_reader: IAreaReader = Depends(get_area_reader),
    gateway: ISentinelGateway = Depends(get_sentinel_gateway),
    cache: IImageCache = Depends(get_image_cache),
    storage: IImageStorage = Depends(get_image_storage),
) -> list[PreviewTileResponse]:
    try:
        await use_cases.fetch_previews(
            area_id=payload.area_id,
            user_id=current_user_id,
            date_start=payload.date_start,
            date_end=payload.date_end,
            area_access_policy=area_access_policy,
            area_reader=area_reader,
            gateway=gateway,
            cache=cache,
            storage=storage,
        )
        return []
        tiles = await use_cases.get_preview_tiles(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            cache=cache,
        )
        return [to_preview_response(t) for t in tiles]
    except InvalidDateRangeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/save", response_model=list[ImageResponse], status_code=201)
async def save_images(
    payload: SaveImagesRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: IImageRepository = Depends(get_image_repository),
    storage: IImageStorage = Depends(get_image_storage),
    cache: IImageCache = Depends(get_image_cache),
) -> list[ImageResponse]:
    try:
        await use_cases.save_images(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
            storage=storage,
            cache=cache,
        )
        images = await use_cases.get_images(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
        )
        return [to_image_response(img) for img in images]
    except NoPreviewAvailableError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("", response_model=list[ImageResponse])
async def get_images(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: IImageRepository = Depends(get_image_repository),
) -> list[ImageResponse]:
    try:
        images = await use_cases.get_images(
            area_id=area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
        )
        return [to_image_response(img) for img in images]
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/{image_id}/png")
async def get_image_png(
    image_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: IImageRepository = Depends(get_image_repository),
    storage: IImageStorage = Depends(get_image_storage),
) -> FastAPIResponse:
    try:
        png_bytes = await use_cases.get_image_as_png(
            image_id=image_id,
            user_id=current_user_id,
            repo=repo,
            area_access_policy=area_access_policy,
            storage=storage,
        )
        return FastAPIResponse(content=png_bytes, media_type="image/png")
    except ImageNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("", status_code=204)
async def delete_images(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: IImageRepository = Depends(get_image_repository),
    storage: IImageStorage = Depends(get_image_storage),
    publisher: IEventPublisher = Depends(get_event_publisher),
    session: AsyncSession = Depends(get_session),
) -> Response:
    try:
        await use_cases.delete_images(
            area_id=area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
            storage=storage,
            publisher=publisher
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    await session.commit()
    await publisher.run_post_commit()
    return Response(status_code=204)

