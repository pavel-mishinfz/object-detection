import io
import os
import shutil
from pathlib import Path
from uuid import UUID

import PIL.Image

from app.image.application.interfaces import IImageStorage


class FileImageStorage(IImageStorage):
    def __init__(self, temp_dir: str, images_dir: str) -> None:
        self._temp_dir = Path(temp_dir)
        self._images_dir = Path(images_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(parents=True, exist_ok=True)

    async def save_temp(self, image_id: UUID, data: bytes) -> None:
        path = self._temp_dir / f"{image_id}.tiff"
        path.write_bytes(data)

    async def promote_to_permanent(self, image_id: UUID) -> str:
        temp_path = self._temp_dir / f"{image_id}.tiff"
        permanent_path = self._images_dir / f"{image_id}.tiff"
        shutil.move(str(temp_path), str(permanent_path))
        return str(permanent_path)

    async def delete(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    async def load_as_png_bytes(self, path: str) -> bytes:
        with PIL.Image.open(path) as img:
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            return buffer.read()

    def get_temp_path(self, image_id: UUID) -> str:
        return str(self._temp_dir / f"{image_id}.tiff")

    def path_exists(self, path: str) -> bool:
        return Path(path).exists()
