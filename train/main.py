import os
import random
from glob import glob
from typing import Tuple, List, Dict, Any, Generator

import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DATASET_NAME = "DeepGlobe"
DATASET_DIR = f"D:/mishinpa/datasets/{DATASET_NAME}/train/"

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

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 30
BATCH_SIZE = 1
INIT_LR = 0.00005
LR_DECREASE_STEP = 15
LR_DECREASE_COEF = 2
PATCH_SIZE = 2432


preprocessing = T.Compose([
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_subimages_generator(
    image: Image.Image,
    subimage_size: Tuple[int, int]
) -> Generator[Image.Image, None, None]:
  for r in range(image.size[1] // subimage_size[1]):
    for c in range(image.size[0] // subimage_size[0]):
      yield image.crop(box=(
              c * subimage_size[0],
              r * subimage_size[1],
              (c + 1) * subimage_size[0],
              (r + 1) * subimage_size[1]
          )
      )


def save_dataset_subimages():
    images_paths = glob(os.path.join(DATASET_DIR, "*_sat.jpg"))
    masks_paths = glob(os.path.join(DATASET_DIR, "*_mask.png"))

    for i, (image_path, mask_path) in enumerate(zip(images_paths, masks_paths)):
        image = Image.open(image_path).crop(box=(8, 8, 2448 - 8, 2448 - 8))
        image_labeled = Image.open(mask_path).crop(box=(8, 8, 2448 - 8, 2448 - 8))

        image.save(fp=f'dataset_{PATCH_SIZE}/originals/image_{i}.jpg')
        image_labeled.save(fp=f'dataset_{PATCH_SIZE}/labeleds/image_labeled_{i}.png')
        # subimages = get_subimages_generator(image=image, subimage_size=(PATCH_SIZE, PATCH_SIZE))
        # subimages_labeleds = get_subimages_generator(image=image_labeled, subimage_size=(PATCH_SIZE, PATCH_SIZE))
        #
        # for si, subimage in enumerate(subimages):
        #     subimage_labeled = next(subimages_labeleds)
        #
        #     subimage.save(fp=f'dataset_{PATCH_SIZE}/originals/i{i}si{si}.jpg')
        #     subimage_labeled.save(fp=f'dataset_{PATCH_SIZE}/labeleds/i{i}si{si}_labeled.png')


def get_image_mask_from_labeled(
    image_labeled: Image.Image,
    classes: Dict[Tuple[int, int, int], Tuple[int, str]]
) -> np.ndarray:

    image_mask = np.zeros(shape=(len(classes), image_labeled.size[0], image_labeled.size[1]))

    image_labeled_ndarray = np.array(object=image_labeled)
    for r in np.arange(stop=image_labeled_ndarray.shape[0]):
        for c in np.arange(stop=image_labeled_ndarray.shape[1]):
            class_rgb = tuple(image_labeled_ndarray[r][c])
            class_value = classes.get(class_rgb)
            if class_value != None:
                image_mask[class_value[0]][r][c] = 1.0
            else:
                image_mask[0][r][c] = 1.0

    return image_mask


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


def transform_pair_images_to_tensor(
    image: Image.Image,
    image_labeled: Image.Image,
    classes: Dict[Tuple[int, int, int], Tuple[int, str]],
    dtype: torch.FloatType = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    image_tensor = T.ToTensor()(pic=image)
    image_mask_tensor = torch.tensor(
        data=get_image_mask_from_labeled(
            image_labeled=image_labeled,
            classes=classes
        ),
        dtype=dtype
    )

    return image_tensor, image_mask_tensor


def reverse_normalize(img, mean, std):
    img = img * np.array(std) + np.array(mean)
    return img


class DeepGlobeDataset(Dataset):
    def __init__(
        self,
        images_paths: str,
        masks_paths: str,
        classes: Dict[Tuple[int, int, int], Tuple[int, str]],
        preprocessing: Any = None
    ):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.classes = classes
        self.preprocessing = preprocessing

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images_paths[idx]).convert('RGB')
        image_labeled = Image.open(self.masks_paths[idx])

        image_tensor, mask_tensor = transform_pair_images_to_tensor(
            image=image, image_labeled=image_labeled, classes=self.classes
        )

        if self.preprocessing:
            image_tensor = self.preprocessing(image_tensor)

        return image_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.images_paths)


# save_dataset_subimages()

images_paths = glob(f'dataset_{PATCH_SIZE}/originals/*')
masks_paths = glob(f'dataset_{PATCH_SIZE}/labeleds/*')

train_valid_images, test_images, train_valid_masks, test_masks = train_test_split(
    images_paths, masks_paths, test_size=0.2
)

train_images, valid_images, train_masks, valid_masks = train_test_split(
    train_valid_images, train_valid_masks, test_size=0.2
)

# Вывод размеров выборок
print(f"Train: {len(train_images)} изображений")
print(f"Valid: {len(valid_images)} изображений")
print(f"Test: {len(test_images)} изображений")

# Использование в вашем классе DeepGlobeDataset
train_dataset = DeepGlobeDataset(
    images_paths=train_images, masks_paths=train_masks,
    classes=CLASSES, preprocessing=preprocessing
)
valid_dataset = DeepGlobeDataset(
    images_paths=valid_images, masks_paths=valid_masks,
    classes=CLASSES, preprocessing=preprocessing
)
test_dataset = DeepGlobeDataset(
    images_paths=test_images, masks_paths=test_masks,
    classes=CLASSES, preprocessing=preprocessing
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=1,
    shuffle=False
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False
)

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

loss = utils.losses.CrossEntropyLoss()
metrics = [
    utils.metrics.Fscore(),
    utils.metrics.IoU()
]
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=INIT_LR),
])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

max_score = 0

loss_logs = {"train": [], "val": []}
metric_logs = {"train": [], "val": []}
for i in range(0, EPOCHS):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_dataloader)
    train_loss, train_metric, train_metric_IOU = list(train_logs.values())
    loss_logs["train"].append(train_loss)
    metric_logs["train"].append(train_metric_IOU)

    valid_logs = valid_epoch.run(valid_dataloader)
    val_loss, val_metric, val_metric_IOU = list(valid_logs.values())
    loss_logs["val"].append(val_loss)
    metric_logs["val"].append(val_metric_IOU)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, 'models/deepglobe_unet_v2.pth')
        print('Model saved!')

    print("LR:", optimizer.param_groups[0]['lr'])
    if i > 0 and i % LR_DECREASE_STEP == 0:
        print('Decrease decoder learning rate')
        optimizer.param_groups[0]['lr'] /= LR_DECREASE_COEF


fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].plot(loss_logs["train"], label = "train")
axes[0].plot(loss_logs["val"], label = "val")
axes[0].set_title("losses - Dice")

axes[1].plot(metric_logs["train"], label = "train")
axes[1].plot(metric_logs["val"], label = "val")
axes[1].set_title("IOU")

[ax.legend() for ax in axes]
plt.tight_layout()
plt.show()


model = torch.load('models/deepglobe_unet_v2.pth', weights_only=False)
model = model.to(DEVICE)
model.eval()


with torch.no_grad():
    # Берем случайный элемент из test_dataset
    n = np.random.choice(len(test_dataset))

    image, mask = test_dataset[n]
    image = image.unsqueeze(0).to(DEVICE)  # Добавляем batch dimension
    mask = mask.numpy()  # Преобразуем маску в numpy для визуализации

    # Получаем предсказание
    pred = model(image)
    pred = pred.cpu().numpy()[0]  # Преобразуем в numpy

    fig, ax = plt.subplots(ncols=3)
    image = image.cpu().numpy()[0].transpose(1, 2, 0)
    image = reverse_normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ax[0].imshow(image)
    ax[1].imshow(
        get_image_labeled_from_mask(
            image_mask=mask,
            classes_by_id=CLASSES_BY_ID
        )
    )
    ax[2].imshow(
        get_image_labeled_from_mask(
            image_mask=pred,
            classes_by_id=CLASSES_BY_ID
        )
    )
    plt.show()