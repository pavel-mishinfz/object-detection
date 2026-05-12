import io
import os
import uuid
from pathlib import Path
from uuid import UUID

import PIL.Image
import rasterio

from app.image.application.interfaces import IImageStorage
from app.image.domain.image import ImageBounds
from app.shared.errors import NotFoundError


class FileImageStorage(IImageStorage):
    def __init__(self, temp_dir: str, images_dir: str) -> None:
        self._temp_dir = Path(temp_dir)
        self._images_dir = Path(images_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(parents=True, exist_ok=True)

    def _make_temp_filename(self, area_id: UUID, image_id: UUID) -> str:
        return f"{area_id}_{image_id}.tiff"

    async def save_temp(self, area_id: UUID, image_id: UUID, data: bytes) -> None:
        path = self._temp_dir / self._make_temp_filename(area_id, image_id)
        path.write_bytes(data)

    async def list_temp_by_area(self, area_id: UUID) -> list[UUID]:
        prefix = f"{area_id}_"
        result = []
        for p in self._temp_dir.glob(f"{prefix}*.tiff"):
            stem = p.stem  # "{area_id}_{image_id}"
            image_id_str = stem[len(prefix):]
            result.append(uuid.UUID(image_id_str))
        return result

    async def get_temp_bounds(self, image_id: UUID) -> ImageBounds:
        path = self.find_temp_path(image_id)
        if path is None:
            raise NotFoundError(f"Временный файл для изображения {image_id} не найден")
        with rasterio.open(path) as src:
            b = src.bounds
        return ImageBounds(min_lat=b.bottom, min_lon=b.left, max_lat=b.top, max_lon=b.right)

    def find_temp_path(self, image_id: UUID) -> str | None:
        matches = list(self._temp_dir.glob(f"*_{image_id}.tiff"))
        return str(matches[0]) if matches else None

    async def promote_to_permanent(self, image_id: UUID) -> str:
        temp_path = self.find_temp_path(image_id)
        if temp_path is None:
            raise NotFoundError(f"Временный файл для изображения {image_id} не найден")
        permanent_path = self._images_dir / f"{image_id}.tiff"
        os.rename(temp_path, str(permanent_path))
        return str(permanent_path)

    async def delete(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            raise NotFoundError(f"Файл по пути {path} не найден")

    async def load_as_png_bytes(self, path: str) -> bytes:
        with PIL.Image.open(path) as img:
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            return buffer.read()
