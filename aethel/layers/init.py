"""Centralized Weight Initialization for Aethel Vision Framework.

Supports Kaiming, Xavier, Normal, and bias zeroing across Conv2d, Linear, and BatchNorm layers.
"""

from typing import Any

try:
    import torch
    import torch.nn as nn
except Exception:
    nn = None  # type: ignore

from aethel.utils.logger import get_logger

logger = get_logger("aethel.layers.init")


def initialize_weights(model: Any, mode: str = "kaiming") -> None:
    """Recursively initialize weights across all module layers in model.

    Args:
        model: PyTorch torch.nn.Module instance.
        mode: Weight initialization strategy ('kaiming', 'xavier', 'normal').
    """
    if nn is None or not hasattr(model, "modules"):
        return

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if mode == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif mode == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif mode == "normal":
                nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    logger.debug(f"Initialized model weights using '{mode}' initialization scheme.")


__all__ = ["initialize_weights"]
