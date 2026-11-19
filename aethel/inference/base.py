"""Abstract Base Predictor / Inference Class for Aethel Vision Framework."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union


class BasePredictor(ABC):
    """Abstract base class for inference engines and real-time predictors."""

    def __init__(self, model_path: Union[str, Path, Any] = "", device: str = "cpu", **kwargs: Any) -> None:
        """Initialize BasePredictor.

        Args:
            model_path: Path to model weights checkpoint (.pt) or exported engine binary (.onnx, .engine).
            device: Target execution device ('cpu', 'cuda').
        """
        self.model_path = Path(model_path) if isinstance(model_path, (str, Path)) and model_path else model_path
        self.device = device

    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Run single image inference and return detection predictions.

        Args:
            image: Input image array (H, W, C) in BGR/RGB format.

        Returns:
            List of prediction dictionary records containing boxes, scores, and class_ids.
        """
        pass

    @abstractmethod
    def predict_batch(self, images: List[Any], **kwargs: Any) -> List[List[Dict[str, Any]]]:
        """Run batch image inference."""
        pass


__all__ = ["BasePredictor"]
