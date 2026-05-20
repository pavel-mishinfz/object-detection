import io
import os
import shutil
import uuid
from pathlib import Path
from uuid import UUID

import numpy as np
import PIL.Image
import rasterio

from app.image.entity.image import ImageBounds
from app.image.exceptions import ImageNotFoundError
from app.image.interfaces.image_storage import IImageStorage


class LocalImageStorage(IImageStorage):
    def __init__(self, temp_dir: str, images_dir: str) -> None:
        self._temp_dir = Path(temp_dir)
        self._images_dir = Path(images_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(parents=True, exist_ok=True)

    async def save_temp(self, image_id: UUID, data: bytes) -> None:
        path = self._build_temp_path(image_id)
        path.write_bytes(data)

    def get_temp_path(self, image_id: UUID) -> str:
        return str(self._build_temp_path(image_id))

    async def move_to_permanent_storage(self, image_id: UUID) -> None:
        temp_path = self._build_temp_path(image_id)
        permanent_path = self._build_permanent_path(image_id)
        shutil.move(str(temp_path), str(permanent_path))
    
    async def get_permanent_path(self, image_id: UUID) -> str:
        return str(self._build_permanent_path(image_id))

    async def delete(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            print(f"Файл по пути {path} не найден")
            #raise ImageNotFoundError(f"Файл по пути {path} не найден")

    async def get_image_as_png_bytes(self, path: str) -> bytes:
        with rasterio.open(path) as src:
            tags = src.tags()
            h = int(tags.get("original_height", src.height))
            w = int(tags.get("original_width", src.width))
            data = src.read()[:, :h, :w]
        img = PIL.Image.fromarray(np.transpose(data, (1, 2, 0)).astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.read()
    
    def _build_temp_path(self, image_id: UUID) -> Path:
        return self._temp_dir / f"{image_id}.tiff"
        
    def _build_permanent_path(self, image_id: UUID) -> Path:
        return self._images_dir / f"{image_id}.tiff"
