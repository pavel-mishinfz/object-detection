import json
import uuid
from uuid import UUID

import redis as redis_lib

from app.image.application.interfaces import IImageCache, PreviewTile
from app.image.domain.image import ImageBounds

_TTL_SECONDS = 86400  # 24 часа


class RedisImageCache(IImageCache):
    def __init__(self, host: str, port: int, db: int) -> None:
        self._redis = redis_lib.Redis(host=host, port=port, db=db, decode_responses=True)

    def _key(self, area_id: UUID) -> str:
        return f"image:preview:{area_id}"

    async def get(self, area_id: UUID) -> tuple[str, list[PreviewTile]] | None:
        raw = self._redis.get(self._key(area_id))
        if raw is None:
            return None
        data = json.loads(raw)
        tiles = [
            PreviewTile(
                image_id=uuid.UUID(t["image_id"]),
                bounds=ImageBounds(
                    min_lat=t["bounds"]["min_lat"],
                    min_lon=t["bounds"]["min_lon"],
                    max_lat=t["bounds"]["max_lat"],
                    max_lon=t["bounds"]["max_lon"],
                ),
            )
            for t in data["tiles"]
        ]
        return data["hash"], tiles

    async def set(
        self,
        area_id: UUID,
        request_hash: str,
        tiles: list[PreviewTile],
    ) -> None:
        data = {
            "hash": request_hash,
            "tiles": [
                {
                    "image_id": str(t.image_id),
                    "bounds": {
                        "min_lat": t.bounds.min_lat,
                        "min_lon": t.bounds.min_lon,
                        "max_lat": t.bounds.max_lat,
                        "max_lon": t.bounds.max_lon,
                    },
                }
                for t in tiles
            ],
        }
        await self._redis.set(self._key(area_id), json.dumps(data), ex=_TTL_SECONDS)

    async def invalidate(self, area_id: UUID) -> None:
        await self._redis.delete(self._key(area_id))
