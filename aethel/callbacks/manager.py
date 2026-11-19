"""Callback Manager Subsystem for Aethel Vision Framework.

Dispatches training lifecycle events to registered callback instances.
"""

from typing import Any, Dict, List, Optional

from aethel.callbacks.base import BaseCallback
from aethel.utils.logger import get_logger

logger = get_logger("aethel.callbacks.manager")


class CallbackManager:
    """Orchestrator for managing and dispatching events to a list of callbacks."""

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None) -> None:
        self.callbacks: List[BaseCallback] = callbacks or []

    def add_callback(self, callback: BaseCallback) -> None:
        """Register a callback instance."""
        self.callbacks.append(callback)

    def on_train_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics)

    def on_batch_start(self, trainer: Any, batch_idx: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_start(trainer, batch_idx)

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, loss)

    def on_validation_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_validation_start(trainer)

    def on_validation_end(self, trainer: Any, val_metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_validation_end(trainer, val_metrics)

    def on_checkpoint_saved(self, trainer: Any, checkpoint_path: str) -> None:
        for cb in self.callbacks:
            cb.on_checkpoint_saved(trainer, checkpoint_path)


__all__ = ["CallbackManager"]
