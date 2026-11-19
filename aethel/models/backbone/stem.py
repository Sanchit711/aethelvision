"""Detector Stem Module for Aethel Vision Backbone.

Executes initial 4x spatial downsampling over RGB input images.
"""

from typing import Any

try:
    import torch
    import torch.nn as nn
    _Module = nn.Module
except Exception:
    class _Module:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

from aethel.layers.conv import Conv


class DetectorStem(_Module):  # type: ignore
    """Detector stem performing 4x downsampling (B, 3, H, W) -> (B, c_stem, H/4, W/4)."""

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        act: Any = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = Conv(in_channels, stem_channels // 2, kernel_size=3, stride=2, act=act, norm=norm)
        self.conv2 = Conv(stem_channels // 2, stem_channels, kernel_size=3, stride=2, act=act, norm=norm)

    def forward(self, x: Any) -> Any:
        return self.conv2(self.conv1(x))


__all__ = ["DetectorStem"]
