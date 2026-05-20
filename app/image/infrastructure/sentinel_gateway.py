import asyncio
from datetime import date
import math
import uuid

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.windows import Window
from sentinelhub import (
    BBox, BBoxSplitter, CRS, DataCollection,
    MimeType, MosaickingOrder, SentinelHubDownloadClient, SentinelHubRequest, SHConfig,
    bbox_to_dimensions,
)

from app.image.dto import TileResult
from app.image.entity.image import ImageBounds
from app.image.interfaces.sentinel_gateway import ISentinelGateway


_RESOLUTION = 10
_MAX_REQUEST_PX = 2500
_TILE_PX = 512
_MAX_CLOUD_COVER = 0.2
_CDSE_COLLECTION = DataCollection.SENTINEL2_L2A.define_from(
    name="sentinel-2-l2a",
    service_url="https://sh.dataspace.copernicus.eu"
)

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
        self._config = self._make_config(client_id, client_secret)

    def _make_config(self, client_id: str, client_secret: str) -> SHConfig:
        config = SHConfig()
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
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
        bbox_list = self._split_bbox(self._build_bbox(coordinates))
        requests = [self._build_request(b, date_start, date_end) for b in bbox_list]
        download_list = [r.get_download_list()[0] for r in requests]

        client = SentinelHubDownloadClient(config=self._config)
        responses = await asyncio.to_thread(
            client.download,
            download_list,
            decode_data=False,
        )

        results = []
        for response in responses:
            results.extend(self._split_into_tiles(response.content))
        return results

    def _build_bbox(self, coordinates: tuple[tuple[float, float], ...]) -> BBox:
        lons = [c[0] for c in coordinates]
        lats = [c[1] for c in coordinates]

        return BBox(
            bbox=(min(lons), min(lats), max(lons), max(lats)),
            crs=CRS.WGS84,
        )

    def _split_bbox(self, bbox: BBox) -> list[BBox]:
        width, height = bbox_to_dimensions(bbox, resolution=_RESOLUTION)
        if width > _MAX_REQUEST_PX or height > _MAX_REQUEST_PX:
            split_x = math.ceil(width / _MAX_REQUEST_PX)
            split_y = math.ceil(height / _MAX_REQUEST_PX)
            splitter = BBoxSplitter([bbox], CRS.WGS84, split_shape=(split_x, split_y))
            return splitter.get_bbox_list()
        return [bbox]

    def _build_request(self, tile_bbox: BBox, date_start: date, date_end: date) -> SentinelHubRequest:
        size = bbox_to_dimensions(tile_bbox, resolution=_RESOLUTION)
        return SentinelHubRequest(
            evalscript=_EVALSCRIPT,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=_CDSE_COLLECTION,
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
            config=self._config,
        )

    def _split_into_tiles(self, tiff_bytes: bytes) -> list[TileResult]:
        results = []
        with MemoryFile(tiff_bytes) as memfile:
            with memfile.open() as src:
                for row_off in range(0, src.height, _TILE_PX):
                    for col_off in range(0, src.width, _TILE_PX):
                        win = Window(
                            col_off=col_off,
                            row_off=row_off,
                            width=min(_TILE_PX, src.width - col_off),
                            height=min(_TILE_PX, src.height - row_off),
                        )
                        data = src.read(window=win)
                        _, h, w = data.shape
                        if h < _TILE_PX or w < _TILE_PX:
                            data = np.pad(data, ((0, 0), (0, _TILE_PX - h), (0, _TILE_PX - w)))
                        win_transform = src.window_transform(win)
                        left, bottom, right, top = array_bounds(h, w, win_transform)
                        results.append(TileResult(
                            image_id=uuid.uuid4(),
                            bounds=ImageBounds(
                                min_lat=bottom,
                                min_lon=left,
                                max_lat=top,
                                max_lon=right,
                            ),
                            tiff_bytes=self._write_tiff(data, win_transform, src.crs, h, w),
                        ))
        return results

    def _write_tiff(self, arr: np.ndarray, transform, crs, original_h: int, original_w: int) -> bytes:
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=arr.shape[1],
                width=arr.shape[2],
                count=arr.shape[0],
                dtype=arr.dtype,
                crs=crs,
                transform=transform,
            ) as dataset:
                dataset.write(arr)
                dataset.update_tags(original_height=original_h, original_width=original_w)
            return memfile.read()
