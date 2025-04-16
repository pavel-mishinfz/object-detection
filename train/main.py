# Basic python and ML Libraries
import os
from glob import glob
import numpy as np

import cv2

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations import ToTensorV2

# defining the files directory and testing directory
CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship",
    "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle",
    "windmill", "none"
]

TRAIN_IMAGES_DIR = 'D:/mishinpa/datasets/DIOR/images/train'
TRAIN_LABELS_DIR = 'D:/mishinpa/datasets/DIOR/labels/train'
WIDTH = 800
HEIGHT = 800
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

EPOCHS = 10
BATCH_SIZE = 16


class DiorDataset(Dataset):

    def __init__(self, images_paths, labels_paths, width, height, transforms=None):
        self.transforms = transforms
        self.images_paths = images_paths
        self.labels_paths = labels_paths
        self.height = height
        self.width = width

    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (self.width, self.height), cv2.INTER_AREA)
        image /= 255.0

        boxes = []
        labels = []

        # cv2 image gives size as height x width
        wt = image.shape[1]
        ht = image.shape[0]

        with open(self.labels_paths[idx], 'r') as f:
            for line in f:
                tokens = line.strip().split()

                class_id = int(tokens[0])  # Индекс класса
                x_center = float(tokens[1])  # Нормализованный центр X
                y_center = float(tokens[2])  # Нормализованный центр Y
                width = float(tokens[3])  # Нормализованная ширина
                height = float(tokens[4])  # Нормализованная высота

                # Преобразование нормализованных координат в абсолютные
                # Для исходного изображения
                x_center_abs = x_center * wt
                y_center_abs = y_center * ht
                width_abs = width * wt
                height_abs = height * ht

                # Вычисление [xmin, ymin, xmax, ymax]
                xmin = x_center_abs - width_abs / 2
                ymin = y_center_abs - height_abs / 2
                xmax = x_center_abs + width_abs / 2
                ymax = y_center_abs + height_abs / 2

                # Масштабирование к новому размеру (self.width, self.height)
                xmin_corr = (xmin / wt) * self.width
                ymin_corr = (ymin / ht) * self.height
                xmax_corr = (xmax / wt) * self.width
                ymax_corr = (ymax / ht) * self.height

                # Добавление в списки
                boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
                labels.append(class_id)

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        # iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        # target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=image,
                                     bboxes=target['boxes'],
                                     labels=labels)

            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image, target

    def __len__(self):
        return len(self.images_paths)


def get_object_detection_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return T.ToPILImage()(img).convert('RGB')


# Function to visualize bounding boxes in the image
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


def collate_fn(batch):
    return tuple(zip(*batch))


images_paths = sorted(glob(os.path.join(TRAIN_IMAGES_DIR, "*.jpg")))[:500]
labels_paths = sorted(glob(os.path.join(TRAIN_LABELS_DIR, "*.txt")))[:500]

train_images, test_images, train_labels, test_labels = train_test_split(
    images_paths, labels_paths, test_size=0.2
)

train_dataset = DiorDataset(train_images, train_labels, WIDTH, HEIGHT, transforms=A.Compose([
    ToTensorV2()
]))
test_dataset = DiorDataset(images_paths, labels_paths, WIDTH, HEIGHT, transforms=A.Compose([
    ToTensorV2()
]))

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

num_classes = len(CLASSES)

# get the model using our helper function
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(DEVICE)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.Adam(
#     params, lr=1e-4, weight_decay=0.0005
# )

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# Логи для потерь и метрик
train_loss_log = []
valid_map_log = []
best_map = 0

# Цикл обучения
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0

    # Тренировочный цикл
    for images, targets in train_dataloader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # Прямой проход и вычисление потерь
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Обратное распространение
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

    # Средняя потеря за эпоху
    avg_train_loss = total_loss / num_batches
    train_loss_log.append(avg_train_loss)

    # Сохранение модели
    torch.save(model.state_dict(), 'models/faster_rcnn_dior.pth')

    # Обновление learning rate
    lr_scheduler.step()

    print(f"Train Loss: {avg_train_loss:.4f}")

# pick one image from the test set
img, target = test_dataset[10]
# put the model in evaluation mode
model.load_state_dict(torch.load('models/faster_rcnn_dior.pth'))
model.eval()
with torch.no_grad():
    prediction = model([img.to(DEVICE)])[0]

nms_prediction = apply_nms(prediction, iou_thresh=0.01)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_img_bbox(torch_to_pil(img), target, ax1, title="Expected Output")
plot_img_bbox(torch_to_pil(img), nms_prediction, ax2, title="Model Output")
plt.tight_layout()
plt.show()