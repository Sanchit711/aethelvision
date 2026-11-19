"""Base Model Interfaces for Aethel Vision Framework.

Defines the fundamental abstract BaseModel architecture from which all proprietary
detectors and backbone vision networks inherit.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore

    class _Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def eval(self) -> "_Module":
            return self

        def train(self, mode: bool = True) -> "_Module":
            return self

        def parameters(self) -> list:
            return []

        def to(self, device: Any) -> "_Module":
            return self

        def state_dict(self, *args: Any, **kwargs: Any) -> dict:
            return {}

        def load_state_dict(self, state_dict: dict, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    nn = type("nn", (), {"Module": _Module})()  # type: ignore


class BaseModel(nn.Module, ABC):
    """Abstract Base Class for all deep learning models in Aethel Vision."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config = config or {}

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Execute forward pass computation."""
        pass

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Compute parameter count."""
        if torch is None or not hasattr(self, "parameters"):
            return 0
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


__all__ = ["BaseModel"]
