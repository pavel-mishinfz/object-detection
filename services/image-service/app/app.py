import io
import uuid
import math
import base64
import json
import cv2
import PIL
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from shapely.geometry import shape

import redis

from .database import DB_INITIALIZER, get_async_session
from .schemas import PolygonMeta, ImageIn, Image
from . import config
from . import crud


cfg: config.Config = config.load_config()

redis_client = redis.Redis(
    host=cfg.redis_host,
    port=cfg.redis_port,
    db=cfg.redis_db
)

app = FastAPI(title='Image Service')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка учетных данных
sh_config = SHConfig()
sh_config.sh_client_id = cfg.client_id
sh_config.sh_client_secret = cfg.client_secret.get_secret_value()
sh_config.sh_base_url = "https://services.sentinel-hub.com"


@app.post(
    '/images/preview',
    response_model=list[dict],
    summary='Предпросмотр снимков из Sentinel Hub для указанного полигона',
    tags=['images']
)
async def preview_images(polygon_meta: PolygonMeta):
    previews = []

    redis_key = f"polygon:{str(polygon_meta.id)}"
    if redis_client.exists(redis_key):
        data = redis_client.hgetall(redis_key)
        cached_hash = data[b'hash'].decode('utf-8')

        if cached_hash == polygon_meta.hash:
            cached_previews = json.loads(data[b'previews'].decode('utf-8'))
            return cached_previews

    images = await load_images(polygon_meta)

    for image in images:
        image_pil = PIL.Image.fromarray(image)

        # --- Создание превью (PNG + base64) ---
        preview_buffer = io.BytesIO()
        image_pil.save(preview_buffer, format="PNG")
        preview_buffer.seek(0)
        preview_base64 = base64.b64encode(preview_buffer.read()).decode("utf-8")

        # --- Сохраняем оригинал в TIFF ---
        tiff_buffer = io.BytesIO()
        image_pil.save(tiff_buffer, format="TIFF")
        tiff_buffer.seek(0)
        image_bytes_tiff = tiff_buffer.read()

        image_preview = {
            "image_id": str(uuid.uuid4()),
            "preview": f"data:image/png;base64,{preview_base64}",
            "original": base64.b64encode(image_bytes_tiff).decode("utf-8")
        }
        previews.append(image_preview)

    # Кэшируем данные о полигоне
    data = {
        'hash': polygon_meta.hash,
        'previews': json.dumps(previews)
    }
    redis_client.hset(f"polygon:{polygon_meta.id}", mapping=data)

    return previews


@app.post(
    '/images/save',
    response_model=list[Image],
    summary='Сохранение снимков после предпросмотра',
    tags=['images']
)
async def save_images(
    polygon_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session)
):
    saved_images = []
    data_collection = DataCollection.SENTINEL2_L2A
    data_folder = Path('C:/') / 'sentinel_downloaded'

    redis_key = f"polygon:{str(polygon_id)}"
    if not redis_client.exists(redis_key):
        raise HTTPException(status_code=500, detail="Данных для предпросмотра не найдено")
    data = redis_client.hgetall(redis_key)
    previews = json.loads(data[b'previews'].decode('utf-8'))

    for preview in previews:
        tiff_data = base64.b64decode(preview['original'])
        abs_path = data_folder / (preview['image_id'] + '.tiff')
        with open(abs_path, 'wb') as f:
            f.write(tiff_data)

        redis_client.delete(redis_key)

        image_in = ImageIn(
            area_id=polygon_id,
            source=data_collection.name,
            url=str(abs_path)
        )

        created_image = await crud.create_image(db, image_in)  # сохраняем информацию о снимке в БД
        saved_images.append(created_image)
    return saved_images


@app.get(
    '/images',
    response_model=list[Image],
    summary='Возвращает информацию о снимках из БД для указанного полигона',
    tags=['images']
)
async def get_images(
    polygon_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session)
):
    return await crud.get_images(db, polygon_id)


async def load_images(polygon_meta: PolygonMeta):

    geometry = shape(polygon_meta.geometry_geojson)
    min_x, min_y, max_x, max_y = geometry.bounds

    resolution = polygon_meta.resolution
    bbox = BBox(bbox=(min_y, min_x, max_y, max_x), crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)

    # при необходимости разбиваем полигон на меньшие участки
    bbox_list = await split_bbox(bbox, size, 1024)

    data_collection = DataCollection.SENTINEL2_L2A

    images = []
    for bbox in bbox_list:
        date_start, date_end = polygon_meta.date_start, polygon_meta.date_end
        # Запрашиваем снимок их хранилища Sentinel BHub
        request = await get_sentinel_image(bbox, date_start, date_end, data_collection, resolution)
        data = request.get_data()
        images.append(data[0])
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
        resolution: int = 10
) -> SentinelHubRequest:
    evalscript = await get_evalscript()
    size = bbox_to_dimensions(bbox, resolution=resolution)
    request = SentinelHubRequest(
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
            return [3.5 * sample.B04, 3.5 * sample.B03, 3.5 * sample.B02]; // увеличение яркости
        }
    """
    return evalscript


@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )