import uuid

from fastapi import APIRouter, Depends, HTTPException, Response

from app.analysis.application import use_cases
from app.analysis.application.interfaces import (
    IAreaAccessPolicy,
    ISegmentationEngine,
    ISegmentationResultRepository,
    IImageReader,
    IObjectTypeRepository,
)
from app.analysis.domain.errors import NoImagesError
from app.analysis.presentation.schemas import (
    SegmentationResultResponse,
    RunSegmentationRequest,
    to_segmentation_result_response,
)
from app.composition import (
    get_area_access_policy,
    get_current_user_id,
    get_segmentation_engine,
    get_segmentation_result_repository,
    get_image_reader,
    get_object_type_repository,
)
from app.shared.errors import AccessDeniedError, NotFoundError

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("", response_model=list[SegmentationResultResponse], status_code=201)
async def run_segmentation(
    payload: RunSegmentationRequest,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    image_reader: IImageReader = Depends(get_image_reader),
    engine: ISegmentationEngine = Depends(get_segmentation_engine),
    repo: ISegmentationResultRepository = Depends(get_segmentation_result_repository),
    object_type_repo: IObjectTypeRepository = Depends(get_object_type_repository),
) -> list[SegmentationResultResponse]:
    try:
        await use_cases.run_segmentation(
            area_id=payload.area_id,
            user_id=current_user_id,
            model_name=payload.model_name,
            area_access_policy=area_access_policy,
            image_reader=image_reader,
            engine=engine,
            repo=repo,
            object_type_repo=object_type_repo,
        )
        results = await use_cases.get_segmentation_results(
            area_id=payload.area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
        )
        return [to_segmentation_result_response(r) for r in results]
    except NoImagesError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("", response_model=list[SegmentationResultResponse])
async def get_segmentation_results(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: ISegmentationResultRepository = Depends(get_segmentation_result_repository),
) -> list[SegmentationResultResponse]:
    try:
        results = await use_cases.get_segmentation_results(
            area_id=area_id,
            user_id=current_user_id,
            area_access_policy=area_access_policy,
            repo=repo,
        )
        return [to_segmentation_result_response(r) for r in results]
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("", status_code=204)
async def delete_segmentation_results(
    area_id: uuid.UUID,
    current_user_id: uuid.UUID = Depends(get_current_user_id),
    area_access_policy: IAreaAccessPolicy = Depends(get_area_access_policy),
    repo: ISegmentationResultRepository = Depends(get_segmentation_result_repository),
) -> Response:
    try:
        await use_cases.delete_segmentation_results(
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
