"""Standard Callbacks Implementation for Aethel Vision Framework.

Provides EarlyStoppingCallback, ModelCheckpointCallback, and ProgressLoggerCallback.
"""

from typing import Any, Dict, Optional

from aethel.callbacks.base import BaseCallback
from aethel.registry.registry import CALLBACK_REGISTRY
from aethel.utils.logger import get_logger

logger = get_logger("aethel.callbacks.standard")


@CALLBACK_REGISTRY.register("ProgressLoggerCallback")
class ProgressLoggerCallback(BaseCallback):
    """Callback for formatting and emitting training progress logs."""

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch [{epoch:03d}] {metrics_str}")


@CALLBACK_REGISTRY.register("EarlyStoppingCallback")
class EarlyStoppingCallback(BaseCallback):
    """Callback for stopping training early when metric stops improving."""

    def __init__(self, monitor: str = "val_loss", patience: int = 10, mode: str = "min") -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return

        improved = (current < self.best_value) if self.mode == "min" else (current > self.best_value)
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"EarlyStopping triggered! No improvement in '{self.monitor}' for {self.patience} epochs.")
                if hasattr(trainer, "stop_training"):
                    trainer.stop_training = True


@CALLBACK_REGISTRY.register("ModelCheckpointCallback")
class ModelCheckpointCallback(BaseCallback):
    """Callback for triggering periodic and best checkpoint saving."""

    def __init__(self, monitor: str = "val_loss", mode: str = "min", save_top_k: int = 1) -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return

        improved = (current < self.best_value) if self.mode == "min" else (current > self.best_value)
        if improved:
            self.best_value = current
            if hasattr(trainer, "save_checkpoint"):
                trainer.save_checkpoint(is_best=True)


__all__ = ["ProgressLoggerCallback", "EarlyStoppingCallback", "ModelCheckpointCallback"]
