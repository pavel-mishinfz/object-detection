from typing import Tuple, Dict, Any
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F


class DeepGlobeDataset(Dataset):
    def __init__(
        self,
        images_paths,
        masks_paths,
        classes: Dict[Tuple[int, int, int], Tuple[int, str]],
        preprocessing: Any,
        augmentation: Any = None
    ):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.classes = classes
        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def get_mask_onehot_from_labeled(
            self,
            image_labeled: Image.Image
    ) -> np.ndarray:
        # (H, W, 3) → (C, H, W)
        image_mask = np.zeros(shape=(len(self.classes), image_labeled.size[0], image_labeled.size[1]))
        image_labeled_ndarray = np.array(image_labeled)
        for r in range(image_labeled_ndarray.shape[0]):
            for c in range(image_labeled_ndarray.shape[1]):
                class_rgb = tuple(image_labeled_ndarray[r][c])
                class_value = self.classes.get(class_rgb)
                if class_value is not None:
                    image_mask[class_value[0]][r][c] = 1.0
                else:
                    image_mask[0][r][c] = 1.0
        return image_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images_paths[idx]).convert('RGB')
        image_labeled = Image.open(self.masks_paths[idx])
        mask_onehot = self.get_mask_onehot_from_labeled(image_labeled)

        if self.augmentation:
            augmented = self.augmentation(image=np.array(image), mask=mask_onehot.transpose(1, 2, 0))
            image, mask_onehot = augmented['image'], augmented['mask']
            mask_onehot = mask_onehot.transpose(2, 0, 1)

        image_tensor = self.preprocessing(image)
        mask_tensor = torch.from_numpy(mask_onehot)

        return image_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.images_paths)

