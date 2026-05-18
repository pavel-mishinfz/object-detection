import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.dependencies import (
    get_detection_engine,
    get_detection_result_repository,
    get_image_reader,
    get_object_type_repository,
)
from app.analysis.infrastructure.detection_engine import YoloDetectionEngine
from app.analysis.exceptions import NoImagesError
from app.analysis.infrastructure.repository import DetectionResultRepository, ObjectTypeRepository
from app.analysis.schemas.analysis import (
    DetectionResultResponse,
    RunAnalysisRequest,
    to_detection_result_response,
)
from app.analysis.services import analysis_service
from app.map.dependencies import get_area_access_policy
from app.shared.contracts import IAreaAccessPolicy, IImageReader
from app.shared.database import get_session
from app.shared.exceptions import AccessDeniedError, NotFoundError
from app.user.infrastructure.auth_backend import get_current_user_id

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("", response_model=list[DetectionResultResponse], status_code=201)
async def run_analysis(
    payload: RunAnalysisRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    image_reader: IImageReader = Depends(get_image_reader),
    engine: YoloDetectionEngine = Depends(get_detection_engine),
    repo: DetectionResultRepository = Depends(get_detection_result_repository),
    object_type_repo: ObjectTypeRepository = Depends(get_object_type_repository),
    session: AsyncSession = Depends(get_session)
) -> list[DetectionResultResponse]:
    try:
        await analysis_service.run_analysis(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            image_reader=image_reader,
            engine=engine,
            repo=repo,
            object_type_repo=object_type_repo,
            session=session
        )
        results = await analysis_service.get_results(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
        )
    except NoImagesError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return [to_detection_result_response(r) for r in results]


@router.get("", response_model=list[DetectionResultResponse])
async def get_detection_results(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: DetectionResultRepository = Depends(get_detection_result_repository),
) -> list[DetectionResultResponse]:
    try:
        results = await analysis_service.get_results(
            area_id=area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return [to_detection_result_response(r) for r in results]


@router.delete("", status_code=204)
async def delete_detection_results(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: DetectionResultRepository = Depends(get_detection_result_repository),
    session: AsyncSession = Depends(get_session)
) -> Response:
    try:
        await analysis_service.delete_results(
            area_id=area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
            session=session
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return Response(status_code=204)
