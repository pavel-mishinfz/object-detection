from pathlib import Path

import cv2
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import rasterio.transform as rt
from rasterio.transform import from_bounds, xy

from app.analysis.application.interfaces import ISegmentationEngine, RawContour

_MIN_CONTOUR_AREA = 50  # px²


class UNetSegmentationEngine(ISegmentationEngine):
    def __init__(self, model_dir: str) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_dir = Path(model_dir)
        self._model = smp.Unet(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=3,
            classes=7,
        )
        
    def load(self, model_name: str) -> None:
        state = torch.load(self._model_dir / model_name, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.to(self._device).eval()

    def segment(self, image_path: str) -> list[RawContour]:
        with rasterio.open(image_path) as src:
            orig_transform = src.transform
            h_orig, w_orig = src.height, src.width
            arr = src.read()  # (3, H, W)

        img = np.moveaxis(arr, 0, -1).astype(np.float32)
        resized = cv2.resize(img, (512, 512))

        img = resized.transpose(2, 0, 1).astype(np.float32)
        tensor = (
            torch.from_numpy(img)
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.no_grad():
            logits = self._model(tensor)  # (1, 7, 512, 512)

        class_mask = logits[0].argmax(dim=0).cpu().numpy()  # (512, 512)

        bounds = rt.array_bounds(h_orig, w_orig, orig_transform)
        seg_transform = from_bounds(*bounds, 512, 512)

        return self._contours_to_raw(class_mask, seg_transform)

    def _contours_to_raw(self, class_mask: np.ndarray, transform) -> list[RawContour]:
        results: list[RawContour] = []
        for class_id in range(1, 7):  # 0 = background, пропускаем
            binary = (class_mask == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
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
                geo_ring.append(geo_ring[0])
                results.append(RawContour(
                    geo_polygon=tuple(geo_ring),
                    object_type_id=class_id + 1,  # DB ids: 1=background, 2=urban_land…
                ))
        return results
