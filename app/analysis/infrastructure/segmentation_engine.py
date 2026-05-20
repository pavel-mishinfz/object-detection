import asyncio
from pathlib import Path

import cv2
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.transform import xy

from app.analysis.dto import RawContour
from app.analysis.interfaces.segmentation_engine import ISegmentationEngine

_MIN_CONTOUR_AREA = 50  # px²
_INPUT_SIZE = 512
_BATCH_SIZE = 8


class UNetSegmentationEngine(ISegmentationEngine):
    def __init__(self, model_dir: str) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_dir = model_dir
        self._model = smp.Unet(encoder_name="resnet101", encoder_weights=None, in_channels=3, classes=7)

    async def load_model(self, model_name: str) -> None:
        state = torch.load(Path(self._model_dir) / model_name, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.to(self._device).eval()

    async def segment_batch(self, image_paths: list[str]) -> list[list[RawContour]]:
        return await asyncio.to_thread(self._run_batch, image_paths)

    def _run_batch(self, image_paths: list[str]) -> list[list[RawContour]]:
        output: list[list[RawContour]] = []
        for i in range(0, len(image_paths), _BATCH_SIZE):
            chunk = image_paths[i : i + _BATCH_SIZE]
            output.extend(self._run(chunk))
        return output

    def _run(self, image_paths: list[str]) -> list[list[RawContour]]:
        tensors, metas = [], []
        for path in image_paths:
            with rasterio.open(path) as src:
                orig_transform = src.transform
                tags = src.tags()
                h_orig = int(tags.get("original_height", src.height))
                w_orig = int(tags.get("original_width", src.width))
                arr = src.read()  # (3, H, W)
            img = np.moveaxis(arr, 0, -1).astype(np.float32)
            tensors.append(torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)))
            metas.append((orig_transform, h_orig, w_orig))

        batch = torch.stack(tensors).to(self._device)  # (N, 3, 512, 512)
        with torch.no_grad():
            logits = self._model(batch)  # (N, 7, 512, 512)

        results: list[list[RawContour]] = []
        for i, (orig_transform, h_orig, w_orig) in enumerate(metas):
            class_mask = logits[i].argmax(dim=0).cpu().numpy()[:h_orig, :w_orig]
            results.append(self._mask_to_contours(class_mask, orig_transform))
        return results

    def _mask_to_contours(self, class_mask: np.ndarray, transform) -> list[RawContour]:
        results: list[RawContour] = []
        for class_id in range(1, 7):  # 0 = background, skip
            binary = (class_mask == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < _MIN_CONTOUR_AREA:
                    continue
                cnt = cv2.approxPolyDP(cnt, epsilon=1.5, closed=True)
                if len(cnt) < 3:
                    continue
                geo_ring: list[tuple[float, float]] = []
                for pt in cnt[:, 0]:
                    lon, lat = xy(transform, int(pt[1]), int(pt[0]))
                    geo_ring.append((lon, lat))
                geo_ring.append(geo_ring[0])  # close the ring
                results.append(RawContour(
                    geo_polygon=tuple(geo_ring),
                    object_type_id=class_id + 1,  # DB ids: 1=background, 2=urban_land…
                ))
        return results
