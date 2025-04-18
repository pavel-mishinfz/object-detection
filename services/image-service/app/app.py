import math
import os
from datetime import date

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    BBox,
    CRS,
    MimeType,
    DataCollection,
    MosaickingOrder,
    bbox_to_dimensions,
    BBoxSplitter
)
from shapely import wkb

from .database import DB_INITIALIZER, get_async_session
from .schemas import PolygonMeta, ImageIn, Image
from . import config
from . import crud


cfg: config.Config = config.load_config()

app = FastAPI(title='Image Service')

# Настройка учетных данных
sh_config = SHConfig()
sh_config.sh_client_id = cfg.client_id
sh_config.sh_client_secret = cfg.client_secret.get_secret_value()
sh_config.sh_base_url = "https://services.sentinel-hub.com"


@app.post(
    '/images',
    response_model=list[Image],
    summary='Скачивает снимки для указанного полигона из Sentinel Hub',
    tags=['images']
)
async def load_images(
        polygon_meta: PolygonMeta,
        db: AsyncSession = Depends(get_async_session)
):

    geometry = wkb.loads(polygon_meta.geometry_wkb)
    min_x, min_y, max_x, max_y = geometry.bounds

    resolution = polygon_meta.resolution
    bbox = BBox(bbox=(min_y, min_x, max_y, max_x), crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)

    # при необходимости разбиваем полигон на меньшие участки
    bbox_list = await split_bbox(bbox, size, 1024)

    data_collection = DataCollection.SENTINEL2_L2A
    data_folder = os.path.join(os.path.abspath(""), 'sentinel_downloaded')
    images = []
    for bbox in bbox_list:
        date_start, date_end = polygon_meta.date_start, polygon_meta.date_end
        # Запрашиваем снимок их хранилища Sentinel BHub
        request = await get_sentinel_image(bbox, date_start, date_end, data_collection, resolution, data_folder)
        request.save_data() # сохраняем снимок локально
        abs_path = os.path.join(data_folder, request.get_filename_list()[0])
        image_in = ImageIn(
            area_id=polygon_meta.id,
            source=data_collection.name,
            url=abs_path
        )
        created_image = await crud.create_image(db, image_in) # сохраняем информацию о снимке в БД
        images.append(created_image)
    return images


async def split_bbox(bbox: BBox, size: tuple[int, int], max_size: int = 2500):
    width, height = size
    if width > max_size or height > max_size:
        # Вычисляем количество ячеек сетки
        split_x = math.ceil(width / max_size)
        split_y = math.ceil(height / max_size)
        bbox_splitter = BBoxSplitter([bbox], CRS.WGS84, split_shape=(split_x, split_y))
        bbox_list = bbox_splitter.get_bbox_list()
    else:
        bbox_list = [bbox]
    return bbox_list


async def get_sentinel_image(
        bbox: BBox,
        date_start: date,
        date_end: date,
        data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
        resolution: int = 10,
        data_folder: str = "sentinel_downloaded"
) -> SentinelHubRequest:
    evalscript = await get_evalscript()
    size = bbox_to_dimensions(bbox, resolution=resolution)
    request = SentinelHubRequest(
        data_folder=data_folder,
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=(date_start, date_end),
                mosaicking_order=MosaickingOrder.LEAST_CC,
                maxcc=0.2
            )
        ],

        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox,
        size=size,
        config=sh_config,
    )
    return request


async def get_evalscript() -> str:
    evalscript = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"], // RGB-каналы
                }],
                output: {
                    bands: 3,
                }
            };
        }

        function evaluatePixel(sample) {
            return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02]; // увеличение яркости
        }
    """
    return evalscript


@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )