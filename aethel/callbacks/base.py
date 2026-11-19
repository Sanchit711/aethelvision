"""Abstract Base Callback Class for Aethel Vision Framework.

Provides event hook interfaces for training lifecycle monitoring, metric logging,
checkpoint saving, and early stopping.
"""

from abc import ABC
from typing import Any, Dict


class BaseCallback(ABC):
    """Abstract base class for event listeners and lifecycle callbacks."""

    def on_train_start(self, trainer: Any) -> None:
        """Fired when training starts."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Fired when training completes."""
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Fired at the start of an epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Fired at the end of an epoch."""
        pass

    def on_batch_start(self, trainer: Any, batch_idx: int) -> None:
        """Fired before processing a batch."""
        pass

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        """Fired after processing a batch."""
        pass

    def on_validation_start(self, trainer: Any) -> None:
        """Fired before validation loop starts."""
        pass

    def on_validation_end(self, trainer: Any, val_metrics: Dict[str, float]) -> None:
        """Fired after validation loop completes."""
        pass

    def on_checkpoint_saved(self, trainer: Any, checkpoint_path: str) -> None:
        """Fired when a model checkpoint is saved to disk."""
        pass


__all__ = ["BaseCallback"]
