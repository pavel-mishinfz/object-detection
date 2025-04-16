import json
from typing import Dict, Tuple
from PIL import Image
import torch
import torchvision
import segmentation_models_pytorch as smp
from torchvision import transforms as T
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import DB_INITIALIZER, get_async_session
from . import config
from . import crud
from . import schemas

from pydantic import BaseModel

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


# Функция для инференса
def segmentation_images(image_path, model):
    # Подготовка изображения
    image = preprocess_image(image_path)
    image = image.to(DEVICE)

    # Инференс
    model.eval()
    with torch.no_grad():
        pred = model(image)

    return image, pred


def get_image_labeled_from_mask(
        image_mask: np.ndarray,
        classes_by_id: Dict[int, Tuple[Tuple[int, int, int], str]]
) -> Image.Image:
    image_labeled_ndarray = np.zeros(
        shape=(image_mask.shape[1], image_mask.shape[2], 3),
        dtype=np.uint8
    )

    image_mask_hot = image_mask.argmax(axis=0)
    for r in np.arange(stop=image_mask_hot.shape[0]):
        for c in np.arange(stop=image_mask_hot.shape[1]):
            class_id = image_mask_hot[r][c]
            class_by_id_value = classes_by_id.get(class_id)
            image_labeled_ndarray[r][c] = np.array(object=class_by_id_value[0])

    image_labeled = Image.fromarray(obj=image_labeled_ndarray)
    return image_labeled


def visualize_segmentation(img, pred):
    fig, ax = plt.subplots(ncols=2)
    image = img.cpu().numpy()[0].transpose(1, 2, 0)
    image = reverse_normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ax[0].imshow(image)
    ax[0].set_title("Original Image")

    ax[1].imshow(
        get_image_labeled_from_mask(pred.cpu().numpy()[0], CLASSES_BY_ID)
    )
    ax[1].set_title("Segmentation Image")

    handles = [
        patches.Patch(color=np.array(color) / 255.0, label=name[1])
        for color, name in CLASSES.items()
    ]
    plt.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=8,
        ncol=2,
        frameon=True,
        edgecolor='black'
    )

    # Настройка отступов для освобождения места под легенду
    plt.subplots_adjust(right=0.65, wspace=0.2)
    plt.show()


# Инициализация модели
model = torch.load(MODEL_PATH, weights_only=False)
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
        # db: AsyncSession = Depends(get_async_session)
):
    # if not os.path.exists(request.image_path):
    #     raise HTTPException(status_code=404, detail="Image not found")
    results = []
    for image_path in request.images_paths:
        image, prediction = segmentation_images(image_path, model)
        results.append((image, prediction))

    for result in results[:10]:
        img, pred = result
        visualize_segmentation(img, pred)


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