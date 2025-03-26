import os
import uuid

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    BBox,
    CRS,
    MimeType,
    DataCollection,
    MosaickingOrder,
    bbox_to_dimensions
)
from shapely import wkb

from .database import DB_INITIALIZER, get_async_session
from . import config
from . import crud


cfg: config.Config = config.load_config()

app = FastAPI(title='Image Service')

# Настройка учетных данных
sh_config = SHConfig()
sh_config.sh_client_id = cfg.client_id
sh_config.sh_client_secret = cfg.client_secret.get_secret_value()
sh_config.sh_base_url = "https://services.sentinel-hub.com"

@app.get(
    '/images',
    # response_model=Area,
    summary='Скачивает снимки для указанного полигона из Sentinel Hub',
    tags=['images']
)
async def load_images(
        area_id: uuid.UUID,
        db: AsyncSession = Depends(get_async_session)
):
    wkb_data = await crud.get_coordinates_by_area_id(db, area_id)
    if wkb_data is None:
        raise HTTPException(status_code=404, detail="Полигон не найден")

    geometry = wkb.loads(wkb_data)
    min_x, min_y, max_x, max_y = geometry.bounds

    resolution = 10
    bbox = BBox(bbox=(min_x, min_y, max_x, max_y), crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)

    data_folder = os.path.join(os.path.abspath(""), 'sentinel_downloaded')
    evalscript = await get_evalscript()
    request = SentinelHubRequest(
        data_folder=data_folder,
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=("2025-02-25", "2025-03-26"),
                mosaicking_order=MosaickingOrder.LEAST_CC
            )
        ],

        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=sh_config,
    )
    request.save_data()
    return os.path.join(data_folder, request.get_filename_list()[0])


async def get_evalscript():
    evalscript = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04", "B08"]
                    }],
                    output: {
                        bands: 3
                    }
                };
            }

            function evaluatePixel(sample) {
                return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
            }
        """
    return evalscript


@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )