import torch
import rasterio
from rasterio.transform import xy
from ultralytics import YOLO

from app.analysis.application.interfaces import IDetectionEngine, RawDetection

_CONF_THRESHOLD = 0.45
_NMS_IOU = 0.6


class YoloDetectionEngine(IDetectionEngine):
    def __init__(self, model_path: str) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(model_path)

    def detect(self, image_path: str) -> list[RawDetection]:
        results = self._model(
            image_path,
            device=self._device,
            conf=_CONF_THRESHOLD,
            iou=_NMS_IOU,
            verbose=False,
        )
        detections: list[RawDetection] = []
        with rasterio.open(image_path) as src:
            transform = src.transform
            for r in results:
                for box, score, label in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    geo_polygon = self._pixel_box_to_geo(transform, box.tolist())
                    detections.append(RawDetection(
                        geo_polygon=geo_polygon,
                        score=float(score),
                        object_type_id=int(label) + 1,  # YOLO is 0-indexed; DB ids start at 1
                    ))
        return detections


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
