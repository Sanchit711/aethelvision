"""Abstract Base Learning Rate Scheduler Interface for Aethel Vision Framework."""

from abc import ABC, abstractmethod
from typing import Any


class BaseScheduler(ABC):
    """Abstract base factory for instantiating learning rate schedulers."""

    @abstractmethod
    def build(
        self,
        optimizer: Any,
        epochs: int,
        warmup_epochs: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Instantiate learning rate scheduler.

        Returns:
            Scheduler instance.
        """
        pass


__all__ = ["BaseScheduler"]
