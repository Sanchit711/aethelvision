"""Abstract Base Optimizer Interface for Aethel Vision Framework."""

from abc import ABC, abstractmethod
from typing import Any, Iterable


class BaseOptimizer(ABC):
    """Abstract base factory for instantiating neural network optimizers."""

    @abstractmethod
    def build(
        self,
        params: Iterable[Any],
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs: Any,
    ) -> Any:
        """Instantiate optimizer for model parameters.

        Returns:
            Optimizer instance.
        """
        pass


__all__ = ["BaseOptimizer"]
