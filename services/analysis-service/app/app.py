import json
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fastapi import FastAPI, Depends, HTTPException

from .database import DB_INITIALIZER, get_async_session
from . import config
from . import crud
from . import schemas

from pydantic import BaseModel

cfg: config.Config = config.load_config()

app = FastAPI(title='Result Service')


CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship",
    "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle",
    "windmill", "none"
]

# Параметры
WIDTH = 800
HEIGHT = 800
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_PATH = 'D:/mishinpa/object-detection/services/analysis-service/app/models/faster_rcnn_dior.pth'


# Функция для загрузки модели
def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Функция для предобработки изображения
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32")
    # image = cv2.resize(image, (WIDTH, HEIGHT), cv2.INTER_AREA)
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


# Функция для инференса
def detect_objects(image_path, model, iou_thresh=0.3, score_thresh=0.5):
    # Подготовка изображения
    image = preprocess_image(image_path)
    image = image.to(DEVICE)

    # Инференс
    model.eval()
    with torch.no_grad():
        predictions = model([image])[0]

    # Применение NMS
    predictions = apply_nms(predictions, iou_thresh)

    return T.ToPILImage()(image).convert('RGB'), predictions


def plot_img_bbox(img, target, ax, title=None):
    ax.imshow(img)

    boxes = target['boxes'].cpu().numpy() if isinstance(target['boxes'], torch.Tensor) else target['boxes']

    for box in boxes:
        x, y, x_max, y_max = box[0], box[1], box[2], box[3]
        width, height = x_max - x, y_max - y
        rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)

    if title:
        ax.set_title(title, fontsize=12)

    ax.axis('off')


# Инициализация модели
num_classes = len(CLASSES)
model = get_object_detection_model(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)


class ImageRequest(BaseModel):
    images_paths: list[str]


@app.post(
    '/analysis',
    # response_model=list,
    summary='Анализирует спутниковые снимки',
    tags=['analysis']
)
async def analysis_images(
        request: ImageRequest,
):
    results = []
    for image_path in request.images_paths:
        image, prediction = detect_objects(image_path, model)
        results.append((image, prediction))

    for result in results[:10]:
        img, pred = result
        fig, ax = plt.subplots()
        plot_img_bbox(img, pred, ax, title="Detection")
        plt.tight_layout()
        plt.show()


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
            await crud.upsert_object_type(
                session, schemas.object_type.ObjectTypeUpsert(**object_type)
            )