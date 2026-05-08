import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.application import use_cases
from app.analysis.domain.errors import (
    AreaAccessDeniedError,
    AreaNotFoundError,
    NoImagesError,
)
from app.analysis.infrastructure.area_reader import AreaReader
from app.analysis.infrastructure.detection_engine import YoloDetectionEngine
from app.analysis.infrastructure.image_reader import ImageReader
from app.analysis.infrastructure.repository import DetectionResultRepository, ObjectTypeRepository
from app.analysis.presentation.schemas import (
    DetectionResultResponse,
    RunAnalysisRequest,
    to_detection_result_response,
)
from app.config import load_config

router = APIRouter(prefix="/analysis", tags=["analysis"])
cfg = load_config()

_detection_engine = YoloDetectionEngine(cfg.model_path)


@router.post("", response_model=list[DetectionResultResponse], status_code=201)
async def run_analysis(
    payload: RunAnalysisRequest,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> list[DetectionResultResponse]:
    area_reader = AreaReader(session)
    repo = DetectionResultRepository(session)
    try:
        await use_cases.run_analysis(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_reader=area_reader,
            image_reader=ImageReader(session),
            engine=_detection_engine,
            repo=repo,
            object_type_repo=ObjectTypeRepository(session),
        )
        results = await use_cases.get_results(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_reader=area_reader,
            repo=repo,
        )
        return [to_detection_result_response(r) for r in results]
    except NoImagesError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except AreaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AreaAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("", response_model=list[DetectionResultResponse])
async def get_detection_results(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> list[DetectionResultResponse]:
    try:
        results = await use_cases.get_results(
            area_id=area_id,
            user_id=current_user_id,
            area_reader=AreaReader(session),
            repo=DetectionResultRepository(session),
        )
        return [to_detection_result_response(r) for r in results]
    except AreaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AreaAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("", status_code=204)
async def delete_detection_results(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(...),
    session: AsyncSession = Depends(...),
) -> Response:
    try:
        await use_cases.delete_results(
            area_id=area_id,
            user_id=current_user_id,
            area_reader=AreaReader(session),
            repo=DetectionResultRepository(session),
        )
        return Response(status_code=204)
    except AreaNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AreaAccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
