import json
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from .database import DB_INITIALIZER, get_async_session
from .schemas import AnalysisIn, Analysis, ObjectTypeUpsert
from . import config
from . import crud

import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
import geopandas as gpd
from geoalchemy2.shape import from_shape


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
MODEL_PATH = 'D:/mishinpa/object-detection/services/analysis-service/app/models/deepglobe_unet_v3.pth'


def crop_pil_to_multiple_of_32(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32
    return image.crop((0, 0, new_width, new_height))


# Функция для предобработки изображения
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = crop_pil_to_multiple_of_32(image)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)  # добавляем batch dim


def reverse_normalize(img, mean, std):
    img = img * np.array(std) + np.array(mean)
    return img


def get_segmentation_mask(prediction):
    pred = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    return pred.astype(np.uint8)


def mask_to_polygons(mask, image_path, simplify_tolerance=0.5):
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width

        height = (height // 32) * 32
        width = (width // 32) * 32

        window = Window(0, 0, width, height)
        image = src.read(window=window)

        transform = src.window_transform(window)
        crs = src.crs

        # Проверка соответствия размеров маски и изображения
        if mask.shape != (height, width):
            raise ValueError(f"Размер маски {mask.shape} не соответствует размерам изображения {(height, width)}")

        # Преобразуем маску в полигоны
        results = (
            {'properties': {'class_id': v}, 'geometry': s}
            for s, v in shapes(mask, transform=transform)
        )

        # Создаём GeoDataFrame для обработки геометрии
        gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs)

        # # Упрощаем геометрию, если задано
        # if simplify_tolerance > 0:
        #     gdf['geometry'] = gdf['geometry'].simplify(simplify_tolerance, preserve_topology=True)

        # Формируем список полигонов
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
    image = preprocess_image(image_path)
    image = image.to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(image)

    mask = get_segmentation_mask(pred)
    polygons = mask_to_polygons(mask, image_path)

    return polygons


# Инициализация модели
model = torch.load(MODEL_PATH, weights_only=False)
model.to(DEVICE)


# def visualize_polygons(polygons, image_path):
#     """
#     Визуализирует исходный снимок с наложенными полигонами.
#
#     Args:
#         polygons: List[dict], список полигонов с классами и геометрией.
#         image_path: str, путь к исходному снимку.
#     """
#     # Читаем изображение
#     with rasterio.open(image_path) as src:
#         crs = src.crs or "EPSG:4326"  # Устанавливаем CRS, если отсутствует
#         image = src.read([1, 2, 3]).transpose(1, 2, 0)  # Читаем RGB
#         height, width = image.shape[:2]
#
#     # Создаём GeoDataFrame из полигонов, исключая фон (class_id == 0)
#     gdf = gpd.GeoDataFrame(
#         [{'geometry': p['geometry'], 'class_id': p['class_id']} for p in polygons if p['class_id'] != 0],
#         crs=crs
#     )
#
#     # Проверяем, что GeoDataFrame не пуст
#     if gdf.empty:
#         print("No valid polygons to display (excluding background).")
#         return
#
#     # Визуализация
#     fig, ax = plt.subplots(figsize=(12, 12))
#     # ax.imshow(image)  # Отображаем исходное изображение
#
#     # Настраиваем цвета для каждого класса
#     colors = {
#         1: 'cyan', 2: 'yellow', 3: 'magenta', 4: 'green', 5: 'blue', 6: 'white'
#     }
#
#     # Отображаем полигоны
#     for class_id in gdf['class_id'].unique():
#         class_gdf = gdf[gdf['class_id'] == class_id]
#         class_gdf.plot(
#             ax=ax,
#             color=colors.get(class_id, 'gray'),
#             # alpha=0.5,
#             label=f"Class {class_id} ({CLASSES_BY_ID[class_id][1]})"
#         )
#
#     plt.title("Original Image with Polygons")
#     plt.legend()
#     plt.show()


@app.post(
    '/analysis',
    response_model=list[Analysis],
    summary='Анализирует спутниковые снимки',
    tags=['analysis']
)
async def analysis_images(
        analysis_in: AnalysisIn,
        db: AsyncSession = Depends(get_async_session)
):
    results = []
    for image_id, image_path in zip(analysis_in.images_ids, analysis_in.images_paths):
        polygons = segmentation_images(image_path, model)

        # Сохранение в базу данных
        for poly in polygons:
            result = await crud.analysis.create_analysis(db, image_id, from_shape(poly['geometry']), poly['class_id'])
            results.append(result)

    return results


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