import os
import uuid
import json
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

import rasterio
from rasterio.transform import xy
from shapely.geometry import Polygon
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

# Параметры
WIDTH = cfg.width_image
HEIGHT = cfg.height_image
MODEL_PATH = cfg.model_path
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Функция для загрузки модели
def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Функция для предобработки изображения
def preprocess_image(image_path) -> torch.Tensor:
    stream = open(image_path, 'rb')
    bytes = bytearray(stream.read())
    array = np.asarray(bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32")
    image /= 255.0
    transform = T.Compose([
        T.ToTensor()
    ])
    image_tensor = transform(image)
    return image_tensor


# Функция для применения NMS
def apply_nms(prediction, iou_thresh=0.3, score_thresh=0.1):
    keep = torchvision.ops.nms(prediction['boxes'], prediction['scores'], iou_thresh)

    scores = prediction['scores'][keep]
    mask = scores >= score_thresh
    final_keep = keep[mask]

    final_prediction = {
        'boxes': prediction['boxes'][final_keep],
        'scores': prediction['scores'][final_keep],
        'labels': prediction['labels'][final_keep]
    }
    return final_prediction

# Функция для инференса
def detect_objects(image_path, model, iou_thresh=0.3):
    # Подготовка изображения
    image_tensor = preprocess_image(image_path)
    image_tensor_with_batch_dim = image_tensor.unsqueeze(0).to(DEVICE)

    # Инференс
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor_with_batch_dim)[0]

    # Применение NMS
    prediction = apply_nms(prediction, iou_thresh)

    return prediction


def pixel_to_geo(image_path, box):
    with rasterio.open(image_path) as src:
        transform = src.transform

        x_min, y_min, x_max, y_max = box
        pixel_coords = [
            (x_min, y_min),  # top-left
            (x_max, y_min),  # top-right
            (x_max, y_max),  # bottom-right
            (x_min, y_max),  # bottom-left
        ]

        geo_coords = []
        for x, y in pixel_coords:
            lon, lat = xy(transform, y, x)
            geo_coords.append((lon, lat))

        return geo_coords


def create_geometry(geo_coords):
    return Polygon(geo_coords)


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
        prediction = detect_objects(image_path, model)

        boxes = prediction['boxes']
        scores = prediction['scores']
        object_types_ids = prediction['labels']
        for box, score, object_type_id in zip(boxes, scores, object_types_ids):
            geo_coords = pixel_to_geo(image_path, box)
            geometry_shape = create_geometry(geo_coords)
            geometry_db = from_shape(geometry_shape)

            result = await crud.analysis.create_analysis(
                db, analysis_in.polygon_id, image_id, geometry_db, score, object_type_id
            )
            results.append(result)
    return results


@app.get(
    '/analysis',
    response_model=list[Analysis],
    summary='Возвращает информацию о результе детекции для указанного полигона',
    tags=['analysis']
)
async def get_detection_results(
    polygon_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session)
):
    return await crud.analysis.get_analysis_results(db, polygon_id)


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
    num_classes = len(objects_type)
    model = get_object_detection_model(num_classes + 1)

    abs_path_to_model = os.path.abspath(MODEL_PATH)
    model.load_state_dict(torch.load(abs_path_to_model, map_location=DEVICE))
    model.to(DEVICE)

    # Cохранение модели в состояние приложения
    app.state.model = model