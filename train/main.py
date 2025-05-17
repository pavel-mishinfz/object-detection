import random
from glob import glob
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import albumentations as A

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from dataset import DeepGlobeDataset
from utils import save_dataset_subimages, get_preprocessing, reverse_normalize, get_image_labeled_from_mask, plot_training_curves
from train import train_model

# Исключение случайности экспериментов
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

CURRENT_DIR = Path().resolve()
DATASET_NAME = "DeepGlobe"
DATASET_DIR = f"D:/mishinpa/datasets/{DATASET_NAME}/train/"
MODEL_PATH = CURRENT_DIR / "models"
PATCH_SIZE = 512

CLASSES = {
    (0, 0, 0): (0, 'background'),
    (0, 255, 255): (1, 'urban_land'),
    (255, 255, 0): (2, 'agriculture_land'),
    (255, 0, 255): (3, 'rangeland'),
    (0, 255, 0): (4, 'forest_land'),
    (0, 0, 255): (5, 'water'),
    (255, 255, 255): (6, 'barren_land'),
}
CLASSES_BY_ID = {id: (rgb, name) for rgb, (id, name) in CLASSES.items()}


class Loss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.4, ce_weight=0.6):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.__name__ = 'Loss'

    def forward(self, y_pred, y_true):
        # [B, C, H, W] → [B, H, W]
        y_true_labels = torch.argmax(y_true, dim=1).long()

        dice_loss = self.dice(y_pred, y_true_labels)
        ce_loss = self.ce(y_pred, y_true_labels)

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


def main():
    # save_dataset_subimages(DATASET_DIR, PATCH_SIZE)

    images_paths = glob(str(CURRENT_DIR) + f'/dataset_{PATCH_SIZE}/originals/*')[:400]
    masks_paths = glob(str(CURRENT_DIR) + f'/dataset_{PATCH_SIZE}/labeleds/*')[:400]

    train_valid_images, test_images, train_valid_masks, test_masks = train_test_split(images_paths, masks_paths,
                                                                                      test_size=0.2, random_state=seed)
    train_images, valid_images, train_masks, valid_masks = train_test_split(train_valid_images, train_valid_masks,
                                                                            test_size=0.2, random_state=seed)

    preprocessing = get_preprocessing()

    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-30, 30), p=0.5)
    ])

    train_dataset = DeepGlobeDataset(train_images, train_masks, CLASSES, preprocessing, augmentation)
    valid_dataset = DeepGlobeDataset(valid_images, valid_masks, CLASSES, preprocessing)
    test_dataset = DeepGlobeDataset(test_images, test_masks, CLASSES, preprocessing)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        classes=len(CLASSES),
        activation='softmax2d'
    )
    model.to(device)

    # class_freqs = [0.8, 0.08, 0.51, 0.06, 0.10, 0.02, 0.15]
    # class_weights = [1.0 / f for f in class_freqs]
    # class_weights = torch.FloatTensor(class_weights).to(device)

    alpha = 1
    loss = Loss(weight=None, dice_weight=alpha, ce_weight=1-alpha) # utils.losses.DiceLoss()
    metrics = [utils.metrics.Fscore(), utils.metrics.IoU()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    loss_logs, metric_logs = train_model(
        model, train_loader, valid_loader,
        loss, metrics, optimizer,
        device, epochs=30,
        checkpoint_path=CURRENT_DIR / 'models/deepglobe_unet.pth'
    )

    plot_training_curves(loss_logs, metric_logs)

    # class_names = [name for _, name in CLASSES.values()]
    # freqs = count_class_distribution(train_dataset)
    # plot_class_distribution(freqs, class_names)

    # Визуализация предсказаний
    model = torch.load(CURRENT_DIR / 'models/deepglobe_unet.pth', weights_only=False)
    model.eval()

    n = np.random.choice(len(test_dataset))
    image, mask = test_dataset[n]
    image = image.unsqueeze(0).to(device)
    pred = model(image).cpu().detach().numpy()[0]

    image_np = image.cpu().numpy()[0].transpose(1, 2, 0)
    image_np = reverse_normalize(image_np, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    ax[0].imshow(image_np)
    ax[0].set_title('Original Image', fontsize=14, pad=10)
    ax[0].axis('off')

    # Ground Truth
    gt_rgb = get_image_labeled_from_mask(mask.numpy(), CLASSES_BY_ID)
    ax[1].imshow(gt_rgb)
    ax[1].set_title('Ground Truth', fontsize=14, pad=10)
    ax[1].axis('off')

    # Prediction
    pred_rgb = get_image_labeled_from_mask(pred, CLASSES_BY_ID)
    ax[2].imshow(pred_rgb)
    ax[2].set_title('Prediction', fontsize=14, pad=10)
    ax[2].axis('off')

    # Create legend
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


if __name__ == '__main__':
    main()
