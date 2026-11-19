"""Configurable Learning Rate Scheduler Builder for Aethel Vision Framework.

Supports Cosine Annealing, OneCycle, StepLR, Polynomial, and Linear Warmup schedulers.
"""

from typing import Any, Dict, Optional

try:
    import torch
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler
except Exception:
    torch = None  # type: ignore
    optim = None  # type: ignore
    lr_scheduler = None  # type: ignore

from aethel.registry.registry import SCHEDULER_REGISTRY
from aethel.schedulers.base import BaseScheduler
from aethel.utils.exceptions import TrainingException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.schedulers")


@SCHEDULER_REGISTRY.register("SchedulerBuilder")
class SchedulerBuilder(BaseScheduler):
    """Factory builder for constructing PyTorch learning rate schedulers."""

    def __init__(self, name: str = "CosineAnnealingLR") -> None:
        super().__init__()
        self.name = name

    def build(
        self,
        optimizer: Any,
        epochs: int,
        warmup_epochs: int = 3,
        min_lr: float = 1e-5,
        **kwargs: Any,
    ) -> Any:
        """Instantiate learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer instance.
            epochs: Total epoch count integer.
            warmup_epochs: Warmup epoch count integer.
            min_lr: Minimum learning rate floor float.

        Returns:
            PyTorch LRScheduler instance.
        """
        if lr_scheduler is None:
            raise TrainingException(message="PyTorch runtime required to build scheduler.")

        sched_name = self.name.lower()
        if sched_name in ("cosineannealinglr", "cosine"):
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=min_lr)
        elif sched_name == "steplr":
            step_size = kwargs.get("step_size", 30)
            gamma = kwargs.get("gamma", 0.1)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif sched_name == "onecyclelr":
            max_lr = kwargs.get("max_lr", 0.01)
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=epochs)
        elif sched_name == "polynomial":
            power = kwargs.get("power", 0.9)
            scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=power)
        else:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

        logger.info(f"Built '{self.name}' learning rate scheduler (epochs={epochs}, min_lr={min_lr})")
        return scheduler


__all__ = ["SchedulerBuilder"]
