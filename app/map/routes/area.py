import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.contracts import IEventPublisher
from app.shared.database import get_session
from app.shared.dependencies import get_event_publisher
from app.map.dependencies import get_area_repository
from app.map.exceptions import (
    PolygonLimitExceededError,
    PolygonNameConflictError,
    PolygonNotFoundError,
    PolygonValidationError,
)
from app.map.infrastructure.repository import AreaRepository
from app.map.schemas.area import (
    CreatePolygonRequest,
    PolygonResponse,
    PolygonSummaryResponse,
    UpdatePolygonRequest,
    to_response,
    to_summary_response,
)
from app.map.services import area_service

router = APIRouter(prefix="/areas", tags=["areas"])


@router.post("", response_model=PolygonResponse, status_code=201)
async def create_area(
    payload: CreatePolygonRequest,
    repo: AreaRepository = Depends(get_area_repository),
    session: AsyncSession = Depends(get_session),
) -> PolygonResponse:
    polygon_id = uuid.uuid4()
    try:
        await area_service.create_polygon(
            polygon_id=polygon_id,
            user_id=payload.user_id,
            name=payload.name,
            coordinates=tuple(tuple(p) for p in payload.geometry.coordinates[0]),
            repo=repo,
            session=session,
        )
    except PolygonValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PolygonNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except PolygonLimitExceededError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return to_response(await area_service.get_polygon(polygon_id, repo))


@router.get("/user/{user_id}", response_model=list[PolygonSummaryResponse])
async def get_user_areas(
    user_id: uuid.UUID,
    repo: AreaRepository = Depends(get_area_repository),
) -> list[PolygonSummaryResponse]:
    polygons = await area_service.get_user_polygons(
        user_id=user_id, repo=repo
    )
    return [to_summary_response(p) for p in polygons]


@router.get("/{polygon_id}", response_model=PolygonResponse)
async def get_area(
    polygon_id: uuid.UUID,
    repo: AreaRepository = Depends(get_area_repository),
) -> PolygonResponse:
    polygon = await repo.find_by_id(polygon_id)
    return to_response(polygon)


@router.put("/{polygon_id}", response_model=PolygonResponse)
async def update_area(
    polygon_id: uuid.UUID,
    payload: UpdatePolygonRequest,
    repo: AreaRepository = Depends(get_area_repository),
    session: AsyncSession = Depends(get_session),
) -> PolygonResponse:
    try:
        await area_service.update_polygon(
            polygon_id=polygon_id,
            user_id=payload.user_id,
            name=payload.name,
            coordinates=tuple(tuple(p) for p in payload.geometry.coordinates[0]),
            repo=repo,
            session=session,
        )
    except PolygonValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except PolygonNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PolygonNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return to_response(await area_service.get_polygon(polygon_id, repo))


@router.delete("/{polygon_id}", status_code=204)
async def delete_area(
    polygon_id: uuid.UUID,
    repo: AreaRepository = Depends(get_area_repository),
    publisher: IEventPublisher = Depends(get_event_publisher),
    session: AsyncSession = Depends(get_session),
) -> Response:
    await area_service.delete_polygon(
        polygon_id=polygon_id,
        repo=repo,
        publisher=publisher,
        session=session,
    )
    return Response(status_code=204)
