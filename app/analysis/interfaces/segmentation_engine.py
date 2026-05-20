from abc import ABC, abstractmethod

from app.analysis.dto import RawContour


class ISegmentationEngine(ABC):
    @abstractmethod
    async def load_model(self, model_name) -> None: ...
    
    @abstractmethod
    async def segment_batch(self, image_paths: list[str]) -> list[list[RawContour]]: ...
