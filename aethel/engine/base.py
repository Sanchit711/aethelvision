"""Aethel Core Execution Engine Base Abstractions."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEngine(ABC):
    """Abstract base class for core execution engines."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize engine state with configuration parameters."""
        pass

    @abstractmethod
    def execute(self) -> Any:
        """Execute core engine lifecycle."""
        pass


__all__ = ["BaseEngine"]
