import uuid

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.map.application import use_cases
from app.map.infrastructure.repository import PolygonRepository
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
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> PolygonResponse:
    polygon_id = uuid.uuid4()
    repo = PolygonRepository(session)
    await use_cases.create_polygon(
        polygon_id=polygon_id,
        user_id=current_user_id,
        name=payload.name,
        coordinates=tuple(tuple(p) for p in payload.geometry.coordinates[0]),
        repo=repo,
    )
    return to_response(await use_cases.get_polygon(
        polygon_id=polygon_id,
        user_id=current_user_id,
        repo=repo,
    ))


@router.get("/{polygon_id}", response_model=PolygonResponse)
async def get_area(
    polygon_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> PolygonResponse:
    repo = PolygonRepository(session)
    return to_response(await use_cases.get_polygon(
        polygon_id=polygon_id,
        user_id=current_user_id,
        repo=repo,
    ))


@router.get("/", response_model=list[PolygonSummaryResponse])
async def get_user_areas(
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> list[PolygonSummaryResponse]:
    repo = PolygonRepository(session)
    polygons = await use_cases.get_user_polygons(user_id=current_user_id, repo=repo)
    return [to_summary_response(p) for p in polygons]


@router.put("/{polygon_id}", response_model=PolygonResponse)
async def update_area(
    polygon_id: uuid.UUID,
    payload: UpdatePolygonRequest,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> PolygonResponse:
    repo = PolygonRepository(session)
    await use_cases.update_polygon(
        polygon_id=polygon_id,
        user_id=current_user_id,
        name=payload.name,
        coordinates=tuple(tuple(p) for p in payload.geometry.coordinates[0]),
        repo=repo,
    )
    return to_response(await use_cases.get_polygon(
        polygon_id=polygon_id,
        user_id=current_user_id,
        repo=repo,
    ))


@router.delete("/{polygon_id}", status_code=204)
async def delete_area(
    polygon_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> Response:
    repo = PolygonRepository(session)
    await use_cases.delete_polygon(
        polygon_id=polygon_id,
        user_id=current_user_id,
        repo=repo,
    )
    return Response(status_code=204)
