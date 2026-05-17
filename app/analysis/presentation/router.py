import uuid

from fastapi import APIRouter, Depends, HTTPException, Response

from app.analysis.application import use_cases
from app.analysis.application.interfaces import (
    IAreaAccessPolicy,
    IDetectionEngine,
    IDetectionResultRepository,
    IImageReader,
    IObjectTypeRepository,
)
from app.analysis.domain.errors import NoImagesError
from app.analysis.presentation.schemas import (
    DetectionResultResponse,
    RunAnalysisRequest,
    to_detection_result_response,
)
from app.composition import (
    get_area_access_policy,
    get_current_user_id,
    get_detection_engine,
    get_detection_result_repository,
    get_image_reader,
    get_object_type_repository,
)
from app.shared.exceptions import AccessDeniedError, NotFoundError

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("", response_model=list[DetectionResultResponse], status_code=201)
async def run_analysis(
    payload: RunAnalysisRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    image_reader: IImageReader = Depends(get_image_reader),
    engine: IDetectionEngine = Depends(get_detection_engine),
    repo: IDetectionResultRepository = Depends(get_detection_result_repository),
    object_type_repo: IObjectTypeRepository = Depends(get_object_type_repository),
) -> list[DetectionResultResponse]:
    try:
        await use_cases.run_analysis(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            image_reader=image_reader,
            engine=engine,
            repo=repo,
            object_type_repo=object_type_repo,
        )
        results = await use_cases.get_results(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
        )
        return [to_detection_result_response(r) for r in results]
    except NoImagesError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("", response_model=list[DetectionResultResponse])
async def get_detection_results(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: IDetectionResultRepository = Depends(get_detection_result_repository),
) -> list[DetectionResultResponse]:
    try:
        results = await use_cases.get_results(
            area_id=area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
        )
        return [to_detection_result_response(r) for r in results]
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("", status_code=204)
async def delete_detection_results(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: IDetectionResultRepository = Depends(get_detection_result_repository),
) -> Response:
    try:
        await use_cases.delete_results(
            area_id=area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
        )
        return Response(status_code=204)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
