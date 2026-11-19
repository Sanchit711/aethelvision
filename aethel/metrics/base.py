"""Abstract Base Metric Class for Aethel Vision Framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics in Aethel Vision."""

    def __init__(self) -> None:
        """Initialize BaseMetric."""
        pass

    @abstractmethod
    def update(self, predictions: Any, targets: Any) -> None:
        """Accumulate predictions and targets for metric computation."""
        pass

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """Compute final evaluation metrics summary dictionary.

        Returns:
            Dictionary mapping metric names (e.g. 'mAP@50') to float values.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal accumulator state."""
        pass


__all__ = ["BaseMetric"]
