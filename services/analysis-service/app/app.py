import os
import uuid
import math
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

import torch
from torchvision import transforms as T

import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from skimage.transform import resize
import geopandas as gpd
from geoalchemy2.shape import from_shape

from .database import DB_INITIALIZER, get_async_session
from .schemas import AnalysisIn, Analysis, ObjectTypeUpsert
from . import config
from . import crud


cfg: config.Config = config.load_config()

app = FastAPI(title='Result Service')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = {
    (0, 0, 0): (0, 'background'),               # black
    (0, 255, 255): (1, 'urban_land'),           # cyan
    (255, 255, 0): (2, 'agriculture_land'),     # yellow
    (255, 0, 255): (3, 'rangeland'),            # magenta
    (0, 255, 0): (4, 'forest_land'),            # green
    (0, 0, 255): (5, 'water'),                  # blue
    (255, 255, 255): (6, 'barren_land'),        # white
}

# Индекс класса -> (RGB, имя)
CLASSES_BY_ID = {
    id: (rgb, name) for rgb, (id, name) in CLASSES.items()
}

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_PATH = cfg.model_path


def load_and_resize_geotiff(image_path):
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3])  # только RGB
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
        transform = src.transform
        crs = src.crs

    height, width = image.shape[:2]
    new_height = math.ceil(height / 32) * 32
    new_width = math.ceil(width / 32) * 32

    resized_image = resize(image, (new_height, new_width), preserve_range=True).astype(np.uint8)
    return resized_image, transform, crs, (height, width), (new_height, new_width)


def update_transform(old_transform, old_shape, new_shape):
    scale_y = old_shape[0] / new_shape[0]
    scale_x = old_shape[1] / new_shape[1]
    return old_transform * Affine.scale(scale_x, scale_y)


def preprocess_image(image_array):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(Image.fromarray(image_array))


def reverse_normalize(img, mean, std):
    img = img * np.array(std) + np.array(mean)
    return img


def get_segmentation_mask(prediction):
    pred = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    return pred.astype(np.uint8)


def mask_to_polygons(mask, transform, crs):
    results = (
        {'properties': {'class_id': v}, 'geometry': s}
        for s, v in shapes(mask.astype(np.uint8), transform=transform)
    )
    gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs)
    polygons = []

    for _, row in gdf.iterrows():
        poly = row['geometry']
        class_id = row['class_id']
        if poly.is_valid:
            polygons.append({
                'class_id': int(class_id) + 1,
                'geometry': poly
            })

    return polygons


def segmentation_images(image_path, model):
    image_np, transform, crs, old_shape, new_shape = load_and_resize_geotiff(image_path)
    image_tensor = preprocess_image(image_np).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(image_tensor)

    mask = get_segmentation_mask(pred)
    new_transform = update_transform(transform, old_shape, new_shape)
    polygons = mask_to_polygons(mask, new_transform, crs)

    # show_image_and_mask(image_path, mask)
    return polygons


def show_image_and_mask(image_path, mask):
    def mask_to_rgb(mask):
        h, w = mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, (color, name) in CLASSES_BY_ID.items():
            rgb_mask[mask == class_id] = color
        return rgb_mask

    # Загрузка исходного изображения
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Настройка фигуры
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Отображение исходного изображения
    axes[0].imshow(image_np)
    axes[0].set_title("Исходное изображение")
    axes[0].axis('off')

    # Отображение маски предсказания
    axes[1].imshow(mask_to_rgb(mask))
    axes[1].set_title("Маска сегментации")
    axes[1].axis('off')

    # Легенда
    legend_elements = [Patch(facecolor=np.array(color) / 255,
                             edgecolor='k',
                             label=f'{class_id}: {name}')
                       for color, (class_id, name) in CLASSES.items()]

    plt.legend(handles=legend_elements,
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               title="Classes",
               borderaxespad=0.,
               framealpha=1)

    plt.tight_layout()
    plt.show()


@app.post(
    '/analysis',
    response_model=list[Analysis],
    summary='Анализирует спутниковые снимки',
    tags=['analysis']
)
async def analysis_images(
    request: Request,
    analysis_in: AnalysisIn,
    db: AsyncSession = Depends(get_async_session)
):
    model = request.app.state.model

    results = await crud.analysis.get_analysis_results(db, analysis_in.polygon_id)
    if results.__len__() > 0:
        return results

    results = []
    for image_id, image_path in zip(analysis_in.images_ids, analysis_in.images_paths):
        polygons = segmentation_images(image_path, model)

        # Сохранение в базу данных
        for poly in polygons:
            geometry_db = from_shape(poly['geometry'])
            object_type_id = poly['class_id']

            result = await crud.analysis.create_analysis(
                db, analysis_in.polygon_id, image_id, geometry_db, object_type_id
            )
            results.append(result)

    return results


@app.get(
    '/analysis',
    response_model=list[Analysis],
    summary='Возвращает информацию о результе сегментации для указанного полигона',
    tags=['analysis']
)
async def get_segmentation_results(
    polygon_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session)
):
    return await crud.analysis.get_analysis_results(db, polygon_id)

 
@app.delete(
    '/analysis',
    summary='Удаляет информацию о результате сегментации для указанного полигона',
    tags=['analysis']
)
async def delete_segmentation_results(
    polygon_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session)
):
    return await crud.analysis.delete_analysis_results(db, polygon_id)


@app.on_event("startup")
async def on_startup():
    await DB_INITIALIZER.init_database(
        cfg.postgres_dsn_async.unicode_string()
    )

    objects_type = []
    with open(cfg.default_objects_config_path, encoding="utf-8") as f:
        objects_type = json.load(f)

    async for session in get_async_session():
        for object_type in objects_type:
            await crud.object_type.upsert_object_type(
                session, ObjectTypeUpsert(**object_type)
            )

    # Инициализация модели
    abs_path_to_model = os.path.abspath(MODEL_PATH)
    model = torch.load(abs_path_to_model, weights_only=False, map_location=DEVICE)
    model.to(DEVICE)

    # Cохранение модели в состояние приложения
    app.state.model = model