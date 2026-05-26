from abc import ABC, abstractmethod

from app.analysis.dto import RawDetection


class IDetectionEngine(ABC):
    @abstractmethod
    async def detect_batch(self, image_paths: list[str]) -> list[list[RawDetection]]: ...
