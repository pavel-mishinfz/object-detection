from datetime import date
import math
import uuid

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from sentinelhub import (
    BBox, BBoxSplitter, CRS, DataCollection,
    MimeType, MosaickingOrder, SentinelHubRequest, SHConfig, bbox_to_dimensions,
)

from app.image.application.interfaces import ISentinelGateway, TileResult
from app.image.domain.image import ImageBounds


_RESOLUTION = 10
_MAX_TILE_PX = 640
_MAX_CLOUD_COVER = 0.2

_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["B02", "B03", "B04"] }],
        output: { bands: 3 }
    };
}
function evaluatePixel(sample) {
    return [3.5 * sample.B04, 3.5 * sample.B03, 3.5 * sample.B02];
}
"""


class SentinelHubGateway(ISentinelGateway):
    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret

    def _make_config(self):
        config = SHConfig()
        config.sh_client_id = self._client_id
        config.sh_client_secret = self._client_secret
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        config.sh_token_url = (
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        )
        return config

    async def fetch_tiles(
        self,
        coordinates: tuple[tuple[float, float], ...],
        date_start: date,
        date_end: date,
    ) -> list[TileResult]:
        lons = [c[0] for c in coordinates]
        lats = [c[1] for c in coordinates]
        bbox = BBox(
            bbox=(min(lons), min(lats), max(lons), max(lats)),
            crs=CRS.WGS84,
        )
        config = self._make_config()
        bbox_list = self._split_bbox(bbox)
        results: list[TileResult] = []

        for tile_bbox in bbox_list:
            size = bbox_to_dimensions(tile_bbox, resolution=_RESOLUTION)
            request = SentinelHubRequest(
                evalscript=_EVALSCRIPT,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from(
                            name="sentinel-2-l2a",
                            service_url="https://sh.dataspace.copernicus.eu"
                        ),
                        time_interval=(date_start, date_end),
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                        maxcc=_MAX_CLOUD_COVER,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF)
                ],
                bbox=tile_bbox,
                size=size,
                config=config,
            )
            data = request.get_data()
            tiff_bytes = self._array_to_tiff(data[0], tile_bbox)
            bounds = ImageBounds(
                min_lat=tile_bbox.min_y,
                min_lon=tile_bbox.min_x,
                max_lat=tile_bbox.max_y,
                max_lon=tile_bbox.max_x,
            )
            results.append(TileResult(
                image_id=uuid.uuid4(),
                tiff_bytes=tiff_bytes,
                bounds=bounds,
            ))

        return results

    def _split_bbox(self, bbox) -> list:
        size = bbox_to_dimensions(bbox, resolution=_RESOLUTION)
        width, height = size
        if width > _MAX_TILE_PX or height > _MAX_TILE_PX:
            split_x = math.ceil(width / _MAX_TILE_PX)
            split_y = math.ceil(height / _MAX_TILE_PX)
            splitter = BBoxSplitter([bbox], CRS.WGS84, split_shape=(split_x, split_y))
            return splitter.get_bbox_list()
        return [bbox]

    def _array_to_tiff(self, image_array, bbox) -> bytes:
        arr = np.moveaxis(image_array, -1, 0)  # (H, W, 3) → (3, H, W)
        transform = from_bounds(
            bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y,
            arr.shape[2], arr.shape[1],
        )
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=arr.shape[1],
                width=arr.shape[2],
                count=3,
                dtype=arr.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dataset:
                dataset.write(arr)
            return memfile.read()
