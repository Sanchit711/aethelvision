"""Configurable Optimizer Builder for Aethel Vision Framework.

Instantiates SGD, Adam, AdamW, and RMSprop optimizers with parameter group support.
"""

from typing import Any, Dict, Iterable, Optional

try:
    import torch
    import torch.optim as optim
except Exception:
    torch = None  # type: ignore
    optim = None  # type: ignore

from aethel.optimizers.base import BaseOptimizer
from aethel.registry.registry import OPTIMIZER_REGISTRY
from aethel.utils.exceptions import TrainingException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.optimizers")


@OPTIMIZER_REGISTRY.register("OptimizerBuilder")
class OptimizerBuilder(BaseOptimizer):
    """Factory builder for constructing PyTorch parameter optimizers."""

    def __init__(self, name: str = "AdamW") -> None:
        super().__init__()
        self.name = name

    def build(
        self,
        params: Iterable[Any],
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        momentum: float = 0.9,
        **kwargs: Any,
    ) -> Any:
        """Instantiate optimizer.

        Args:
            params: Model parameter iterable.
            lr: Learning rate float.
            weight_decay: Weight decay penalty float.
            momentum: Momentum factor (for SGD / RMSprop).

        Returns:
            PyTorch optim.Optimizer instance.
        """
        if optim is None:
            raise TrainingException(message="PyTorch runtime required to build optimizer.")

        opt_name = self.name.lower()
        if opt_name == "sgd":
            optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        elif opt_name == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
        elif opt_name == "adamw":
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
        elif opt_name == "rmsprop":
            optimizer = optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)
        else:
            raise TrainingException(
                message=f"Unsupported optimizer algorithm '{self.name}'",
                details="Supported optimizers: SGD, Adam, AdamW, RMSprop",
            )

        logger.info(f"Built '{self.name}' optimizer (lr={lr}, weight_decay={weight_decay})")
        return optimizer


__all__ = ["OptimizerBuilder"]
