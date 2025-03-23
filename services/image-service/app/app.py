import uuid

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from sentinelhub import SHConfig, SentinelHubRequest, BBox, CRS, MimeType, DataCollection, Geometry, bbox_to_dimensions
import matplotlib.pyplot as plt
from shapely import wkb
import numpy as np

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
    geometry = wkb.loads(wkb_data)
    min_x, min_y, max_x, max_y = geometry.bounds

    resolution = 60
    betsiboka_bbox = BBox(bbox=(37.620, 55.755, 37.630, 55.765), crs=CRS.WGS84)
    betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=("2020-06-12", "2025-03-24"),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=betsiboka_bbox,
        size=betsiboka_size,
        config=sh_config,
    )

    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]

    image_bright = image.astype(float) * 1.5  # Умножаем на 1.5 для повышения яркости
    image_bright = np.clip(image_bright, 0, 255).astype(np.uint8)  # Ограничиваем до 0–255

    plt.imsave("output_image.png", image_bright)

    # Отображение
    plt.figure(figsize=(10, 10))
    plt.imshow(image_bright)
    plt.axis('off')
    plt.title("Brightened True Color Image (Scaling)")
    plt.show()


@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )