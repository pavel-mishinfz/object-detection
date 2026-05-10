import uuid

from fastapi import APIRouter, Depends, HTTPException, Response

from app.composition import (
    get_current_user_id,
    get_image_files_cleaner,
    get_polygon_repository,
)
from app.map.application import use_cases
from app.map.application.interfaces import IImageFilesCleaner, IPolygonRepository
from app.map.domain.errors import (
    PolygonAccessDeniedError,
    PolygonLimitExceededError,
    PolygonNameConflictError,
    PolygonNotFoundError,
    PolygonValidationError,
)
from app.map.presentation.schemas import (
    CreatePolygonRequest,
    PolygonResponse,
    PolygonSummaryResponse,
    UpdatePolygonRequest,
    to_response,
    to_summary_response,
)

router = APIRouter(prefix="/areas", tags=["areas"])


@router.post("/", response_model=PolygonResponse, status_code=201)
async def create_area(
    payload: CreatePolygonRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    repo: IPolygonRepository = Depends(get_polygon_repository),
) -> PolygonResponse:
    polygon_id = uuid.uuid4()
    try:
        await use_cases.create_polygon(
            polygon_id=polygon_id,
            user_id=current_user_id,
            name=payload.name,
            coordinates=tuple(tuple(p) for p in payload.geometry.coordinates[0]),
            repo=repo,
        )
    except PolygonValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PolygonNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except PolygonLimitExceededError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return to_response(await use_cases.get_polygon(
        polygon_id=polygon_id,
        user_id=current_user_id,
        repo=repo,
    ))


@router.get("/", response_model=list[PolygonSummaryResponse])
async def get_user_areas(
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    repo: IPolygonRepository = Depends(get_polygon_repository),
) -> list[PolygonSummaryResponse]:
    polygons = await use_cases.get_user_polygons(user_id=current_user_id, repo=repo)
    return [to_summary_response(p) for p in polygons]


@router.get("/{polygon_id}", response_model=PolygonResponse)
async def get_area(
    polygon_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    repo: IPolygonRepository = Depends(get_polygon_repository),
) -> PolygonResponse:
    try:
        return to_response(await use_cases.get_polygon(
            polygon_id=polygon_id,
            user_id=current_user_id,
            repo=repo,
        ))
    except PolygonNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PolygonAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.put("/{polygon_id}", response_model=PolygonResponse)
async def update_area(
    polygon_id: uuid.UUID,
    payload: UpdatePolygonRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    repo: IPolygonRepository = Depends(get_polygon_repository),
) -> PolygonResponse:
    try:
        await use_cases.update_polygon(
            polygon_id=polygon_id,
            user_id=current_user_id,
            name=payload.name,
            coordinates=tuple(tuple(p) for p in payload.geometry.coordinates[0]),
            repo=repo,
        )
    except PolygonValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PolygonNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PolygonAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except PolygonNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return to_response(await use_cases.get_polygon(
        polygon_id=polygon_id,
        user_id=current_user_id,
        repo=repo,
    ))


@router.delete("/{polygon_id}", status_code=204)
async def delete_area(
    polygon_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    repo: IPolygonRepository = Depends(get_polygon_repository),
    image_files_cleaner: IImageFilesCleaner = Depends(get_image_files_cleaner),
) -> Response:
    try:
        await use_cases.delete_polygon(
            polygon_id=polygon_id,
            user_id=current_user_id,
            repo=repo,
            image_files_cleaner=image_files_cleaner,
        )
    except PolygonNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PolygonAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return Response(status_code=204)
