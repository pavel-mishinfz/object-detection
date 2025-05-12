import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoShipDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        # self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [
            img_id for img_id in self.coco.getImgIds()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert("RGB")

        boxes, labels, area, iscrowd = [], [], [], []
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])
            area.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        img_id = torch.tensor([img_id])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)
