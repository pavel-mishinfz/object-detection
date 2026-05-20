import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.contracts import IImageReader
from app.analysis.dependencies import (
    get_image_reader,
    get_object_type_repository,
    get_segmentation_engine,
    get_segmentation_result_repository,
)
from app.analysis.exceptions import NoImagesError
from app.analysis.interfaces.repository import IObjectTypeRepository, ISegmentationResultRepository
from app.analysis.interfaces.segmentation_engine import ISegmentationEngine
from app.analysis.schemas.analysis import (
    RunSegmentationRequest,
    SegmentationResultResponse,
    to_segmentation_result_response,
)
from app.analysis.services import analysis_service
from app.shared.database import get_session
from app.shared.exceptions import AccessDeniedError, NotFoundError

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("", response_model=list[SegmentationResultResponse], status_code=201)
async def run_segmentation(
    payload: RunSegmentationRequest,
    image_reader: IImageReader = Depends(get_image_reader),
    engine: ISegmentationEngine = Depends(get_segmentation_engine),
    repo: ISegmentationResultRepository = Depends(get_segmentation_result_repository),
    object_type_repo: IObjectTypeRepository = Depends(get_object_type_repository),
    session: AsyncSession = Depends(get_session),
) -> list[SegmentationResultResponse]:
    try:
        await analysis_service.run_segmentation(
            area_id=payload.area_id,
            model_name=payload.model_name,
            image_reader=image_reader,
            engine=engine,
            repo=repo,
            object_type_repo=object_type_repo,
            session=session,
        )
        results = await analysis_service.get_segmentation_results(
            area_id=payload.area_id,
            repo=repo,
        )
    except NoImagesError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return [to_segmentation_result_response(r) for r in results]


@router.get("", response_model=list[SegmentationResultResponse])
async def get_segmentation_results(
    area_id: uuid.UUID,
    repo: ISegmentationResultRepository = Depends(get_segmentation_result_repository),
) -> list[SegmentationResultResponse]:
    try:
        results = await analysis_service.get_segmentation_results(
            area_id=area_id,
            repo=repo,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return [to_segmentation_result_response(r) for r in results]


@router.delete("", status_code=204)
async def delete_segmentation_results(
    area_id: uuid.UUID,
    repo: ISegmentationResultRepository = Depends(get_segmentation_result_repository),
    session: AsyncSession = Depends(get_session),
) -> Response:
    try:
        await analysis_service.delete_segmentation_results(
            area_id=area_id,
            repo=repo,
            session=session,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return Response(status_code=204)
