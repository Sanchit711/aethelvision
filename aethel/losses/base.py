"""Abstract Base Loss Class for Aethel Vision Framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

try:
    import torch
    import torch.nn as nn
    _Module = nn.Module
except Exception:
    class _Module:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)


class BaseLoss(_Module, ABC):  # type: ignore
    """Abstract base class for all loss function formulations in Aethel Vision."""

    def __init__(self) -> None:
        """Initialize BaseLoss."""
        super().__init__()

    @abstractmethod
    def forward(
        self,
        predictions: Any,
        targets: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Compute loss between model predictions and ground-truth targets.

        Args:
            predictions: Model output predictions.
            targets: Ground-truth target annotations.

        Returns:
            Tuple of (total_loss_scalar_tensor, breakdown_dictionary).
        """
        pass


__all__ = ["BaseLoss"]
