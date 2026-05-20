import asyncio

import rasterio
import torch
from rasterio.transform import xy
from ultralytics import YOLO

from app.analysis.dto import RawDetection
from app.analysis.interfaces.detection_engine import IDetectionEngine

_CONF_THRESHOLD = 0.45
_NMS_IOU = 0.6
_BATCH_SIZE = 8


class YoloDetectionEngine(IDetectionEngine):
    def __init__(self, model_path: str) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(model_path)

    async def detect_batch(self, image_paths: list[str]) -> list[list[RawDetection]]:
        return await asyncio.to_thread(self._run_inference_batch, image_paths)

    def _run_inference_batch(self, image_paths: list[str]) -> list[list[RawDetection]]:
        output: list[list[RawDetection]] = []
        for i in range(0, len(image_paths), _BATCH_SIZE):
            chunk = image_paths[i : i + _BATCH_SIZE]
            chunk_results = self._model(
                chunk,
                device=self._device,
                conf=_CONF_THRESHOLD,
                iou=_NMS_IOU,
                verbose=False,
            )
            for result, image_path in zip(chunk_results, chunk):
                with rasterio.open(image_path) as src:
                    transform = src.transform
                detections = [
                    RawDetection(
                        geo_polygon=self._pixel_box_to_geo(transform, box.tolist()),
                        score=float(score),
                        object_type_id=int(label) + 1,
                    )
                    for box, score, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
                ]
                output.append(detections)
        return output

    def _pixel_box_to_geo(
        self, transform: rasterio.transform.Affine, box: list[float],
    ) -> tuple[tuple[float, float], ...]:
        x_min, y_min, x_max, y_max = box
        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        geo: list[tuple[float, float]] = []
        for px, py in corners:
            lon, lat = xy(transform, py, px)
            geo.append((lon, lat))
        geo.append(geo[0])  # close the ring
        return tuple(geo)
