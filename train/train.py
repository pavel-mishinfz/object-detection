import os
import json
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from pycocotools.cocoeval import COCOeval

from dataset import CocoShipDataset
from utils import collate_fn, get_transform, plot_metrics


# Исключение случайности экспериментов
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Настройки
VERSION = 11
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2  # 1 класс + фон
BATCH_SIZE = 8
EPOCHS = 50

# Пути
DATA_DIR = Path("D:/mishinpa/datasets/sentinel-2-ship")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"
TRAIN_ANN = TRAIN_DIR / "_annotations.coco.json"
VAL_ANN = VAL_DIR / "_annotations.coco.json"
TEST_ANN = TEST_DIR / "_annotations.coco.json"
MODEL_DIR = Path().resolve() / "models"


def convert_to_coco_format(predictions):
    results = []
    for i, prediction in enumerate(predictions):
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        image_id = prediction["image_id"]

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            result = {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [float(x_min), float(y_min), float(width), float(height)],
                "score": float(score)
            }
            results.append(result)

    return results


def evaluate_model(model, dataloader, device):
    model.eval()
    coco_gt = dataloader.dataset.coco
    coco_results = []

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for i, output in enumerate(outputs):
            output["image_id"] = targets[i]["image_id"].item()
        coco_results.extend(convert_to_coco_format(outputs))

    result_file = "temp_coco_results.json"
    with open(result_file, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    os.remove(result_file)

    stats = coco_eval.stats
    mAP = float(stats[0])  # mAP@[.5:.95]
    ap_50 = float(stats[1])  # AP@0.5

    return {
        "mAP": mAP,
        "AP@0.5": ap_50
    }


# Модель
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Загрузка данных
train_dataset = CocoShipDataset(TRAIN_DIR, TRAIN_ANN, transforms=get_transform())
val_dataset = CocoShipDataset(VAL_DIR, VAL_ANN, transforms=get_transform())
test_dataset = CocoShipDataset(TEST_DIR, TEST_ANN, transforms=get_transform())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = get_model(NUM_CLASSES)
model.to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

best_map = 0.0
best_model_path = MODEL_DIR / f"faster_rcnn_epoch_{EPOCHS}_v{VERSION}.pth"
train_losses, val_metrics = [], []

# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#
#     # Обучение
#     for images, targets in train_loader:
#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
#         total_loss += losses.item()
#
#     lr_scheduler.step()
#
#     avg_loss = total_loss / len(train_loader)
#     train_losses.append(avg_loss)
#
#     metrics = evaluate_model(model, val_loader, DEVICE)
#     val_metrics.append(metrics)
#
#     current_lr = optimizer.param_groups[0]['lr']
#     print(f"🔁 Current LR: {current_lr:.6f}")
#
#     print(f"[Epoch {epoch + 1}] Train loss: {avg_loss:.4f}, Val mAP: {metrics['mAP']:.4f}, AP@0.5: {metrics['AP@0.5']:.4f}")
#
#     # Сохранение лучшей модели
#     if metrics['mAP'] > best_map:
#         best_map = metrics['mAP']
#         torch.save(model.state_dict(), best_model_path)
#         print(f"✔️ Best model saved (mAP: {best_map:.4f})")
#
# plot_metrics(train_losses, val_metrics)
