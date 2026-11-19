"""Class Expansion Subsystem for Continual Learning in Aethel Vision Framework.

Provides ClassExpansionManager to dynamically expand model detection / classification head layers
when new classes are registered, preserving existing weights while initializing new class channels.
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

from aethel.classes.class_manager import ClassManager
from aethel.utils.logger import get_logger

logger = get_logger("aethel.continual.expansion")


class ClassExpansionManager:
    """Manages dynamic head channel expansion for continual learning on new class datasets."""

    def __init__(self, class_manager: Optional[ClassManager] = None) -> None:
        """Initialize ClassExpansionManager.

        Args:
            class_manager: ClassManager taxonomy instance.
        """
        self.class_manager = class_manager or ClassManager()

    def expand_head_conv(
        self,
        conv_layer: Any,
        num_new_classes: int,
        old_num_classes: int,
    ) -> Any:
        """Expand output channels of a 2D Conv layer for new classes while copying existing weights.

        Args:
            conv_layer: Existing nn.Conv2d module.
            num_new_classes: Number of new classes added.
            old_num_classes: Original number of classes.

        Returns:
            New expanded nn.Conv2d module with preserved weights for existing classes.
        """
        if torch is None or nn is None or not hasattr(conv_layer, "weight"):
            return conv_layer

        in_channels = conv_layer.in_channels
        old_out_channels = conv_layer.out_channels
        new_out_channels = old_out_channels + num_new_classes

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=new_out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:old_out_channels] = conv_layer.weight
            if conv_layer.bias is not None:
                new_conv.bias[:old_out_channels] = conv_layer.bias

        logger.info(
            f"Expanded Conv2d head output channels from {old_out_channels} -> {new_out_channels} "
            f"(+{num_new_classes} new classes)"
        )
        return new_conv


__all__ = ["ClassExpansionManager"]
