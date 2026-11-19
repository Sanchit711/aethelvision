"""Abstract Base Visualizer Class for Aethel Vision Framework."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseVisualizer(ABC):
    """Abstract base class for visualizers and plotters in Aethel Vision."""

    @abstractmethod
    def draw_detections(
        self,
        image: Any,
        boxes: Any,
        scores: Any,
        class_ids: Any,
        class_names: Optional[List[str]] = None,
    ) -> Any:
        """Render spatial bounding boxes and class labels onto an image tensor array.

        Returns:
            Annotated image array.
        """
        pass


__all__ = ["BaseVisualizer"]
