import os
import json
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

import rasterio
from shapely.geometry import box
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
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Функция для предобработки изображения
def preprocess_image(image_path) -> torch.Tensor:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32")
    image = cv2.resize(image, (WIDTH, HEIGHT), cv2.INTER_AREA)
    image /= 255.0
    transform = ToTensorV2()
    image = transform(image=image)['image']
    return image


# Функция для применения NMS
def apply_nms(prediction, iou_thresh=0.3):
    keep = torchvision.ops.nms(prediction['boxes'], prediction['scores'], iou_thresh)
    final_prediction = {
        'boxes': prediction['boxes'][keep].cpu().numpy(),
        'scores': prediction['scores'][keep].cpu().numpy(),
        'labels': prediction['labels'][keep].cpu().numpy()
    }
    return final_prediction


def transform_tensor_to_image(tensor):
    return T.ToPILImage()(tensor)


# Функция для инференса
def detect_objects(image_path, model, iou_thresh=0.3):
    # Подготовка изображения
    tensor_image = preprocess_image(image_path)
    tensor_image_with_batch_dim = tensor_image.unsqueeze(0).to(DEVICE)

    # Инференс
    model.eval()
    with torch.no_grad():
        predictions = model(tensor_image_with_batch_dim)[0]

    # Применение NMS
    predictions = apply_nms(predictions, iou_thresh)

    return transform_tensor_to_image(tensor_image), predictions


def pixel_to_geo(image_path, box):
    with rasterio.open(image_path) as src:
        original_height, original_width = src.height, src.width
        transform = src.transform

        scale_x = original_width / WIDTH
        scale_y = original_height / HEIGHT

        xmin, ymin, xmax, ymax = box
        xmin *= scale_x
        xmax *= scale_x
        ymin *= scale_y
        ymax *= scale_y

        # Получаем координаты углов рамки
        lon_min, lat_min = rasterio.transform.xy(transform, ymin, xmin, offset='ul')
        lon_max, lat_max = rasterio.transform.xy(transform, ymax, xmax, offset='lr')

        return [lon_min, lat_min, lon_max, lat_max]


def create_geometry(geo_coords):
    geom = box(geo_coords[0], geo_coords[1], geo_coords[2], geo_coords[3])
    return geom


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

    results = []
    for image_id, image_path in zip(analysis_in.images_ids, analysis_in.images_paths):
        image, prediction = detect_objects(image_path, model)

        boxes = prediction['boxes']
        scores = prediction['scores']
        object_types_ids = prediction['labels']
        for box, score, object_type_id in zip(boxes, scores, object_types_ids):
            geo_coords = pixel_to_geo(image_path, box)
            geometry_shape = create_geometry(geo_coords)
            geometry_db = from_shape(geometry_shape)

            result = await crud.analysis.create_analysis(db, image_id, geometry_db, score, object_type_id)
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

    # Инициализация модели
    num_classes = len(objects_type)
    model = get_object_detection_model(num_classes + 1)

    abs_path_to_model = os.path.abspath(MODEL_PATH)
    model.load_state_dict(torch.load(abs_path_to_model, map_location=DEVICE))
    model.to(DEVICE)

    # Cохранение модели в состояние приложения
    app.state.model = model