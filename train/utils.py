import os
import numpy as np
from typing import Tuple, Generator, Dict
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as T


def get_subimages_generator(image: Image.Image, subimage_size: Tuple[int, int]) -> Generator[Image.Image, None, None]:
    for r in range(image.size[1] // subimage_size[1]):
        for c in range(image.size[0] // subimage_size[0]):
            yield image.crop(box=(
                c * subimage_size[0],
                r * subimage_size[1],
                (c + 1) * subimage_size[0],
                (r + 1) * subimage_size[1]
            ))


def save_dataset_subimages(dataset_dir, patch_size):
    from glob import glob
    images_paths = sorted(glob(os.path.join(dataset_dir, "*_sat.jpg")))
    masks_paths = sorted(glob(os.path.join(dataset_dir, "*_mask.png")))

    os.makedirs(f'dataset_{patch_size}/originals', exist_ok=True)
    os.makedirs(f'dataset_{patch_size}/labeleds', exist_ok=True)

    for i, (image_path, mask_path) in enumerate(zip(images_paths, masks_paths)):
        image = Image.open(image_path).crop(box=(200, 200, 2448 - 200, 2448 - 200))
        image_labeled = Image.open(mask_path).crop(box=(200, 200, 2448 - 200, 2448 - 200))

        subimages = get_subimages_generator(image=image, subimage_size=(patch_size, patch_size))
        subimages_labeleds = get_subimages_generator(image=image_labeled, subimage_size=(patch_size, patch_size))

        for si, subimage in enumerate(subimages):
            subimage_labeled = next(subimages_labeleds)
            subimage.save(fp=f'dataset_{patch_size}/originals/i{i}si{si}.jpg')
            subimage_labeled.save(fp=f'dataset_{patch_size}/labeleds/i{i}si{si}_labeled.png')


def get_preprocessing():
    transforms = [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return T.Compose(transforms)


def get_image_labeled_from_mask(image_mask: np.ndarray, classes_by_id: Dict[int, Tuple[Tuple[int, int, int], str]]) -> Image.Image:
    image_labeled_ndarray = np.zeros(
        shape=(image_mask.shape[1], image_mask.shape[2], 3),
        dtype=np.uint8
    )

    image_mask_hot = image_mask.argmax(axis=0)
    for r in range(image_mask_hot.shape[0]):
        for c in range(image_mask_hot.shape[1]):
            class_id = image_mask_hot[r][c]
            image_labeled_ndarray[r][c] = np.array(classes_by_id[class_id][0])

    return Image.fromarray(image_labeled_ndarray)


def reverse_normalize(img, mean, std):
    img = img * np.array(std) + np.array(mean)
    return img


def plot_training_curves(loss_logs, metric_logs):
    epochs = list(range(1, len(loss_logs['train']) + 1))
    plt.figure(figsize=(10, 4))

    # График функции потерь (Train + Validation)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_logs['train'], "b-o", label="Train")
    plt.plot(epochs, loss_logs['val'], "r--o", label="Validation")  # Добавлено
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # График метрики
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metric_logs['train'], "g-o", label="Train")
    plt.plot(epochs, metric_logs['val'], "m--o", label="Validation")
    plt.title("IoU")
    plt.xlabel("Epoch")
    plt.xlabel("IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

