"""Attention Layer Primitives for Aethel Vision Framework.

Defines the AttentionHook abstract interface and native CBAM (Convolutional Block Attention Module)
combining channel and spatial attention mechanisms.
"""

from abc import ABC, abstractmethod
from typing import Any

try:
    import torch
    import torch.nn as nn
    _Module = nn.Module
except Exception:
    torch = None  # type: ignore
    class _Dummy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return args[0] if args else None
    class _NN:
        Module = _Dummy
        Conv2d = _Dummy
        BatchNorm2d = _Dummy
        Identity = _Dummy
        Sequential = _Dummy
        AdaptiveAvgPool2d = _Dummy
        AdaptiveMaxPool2d = _Dummy
        ReLU = _Dummy
        Sigmoid = _Dummy
    nn = _NN()  # type: ignore
    _Module = nn.Module


class AttentionHook(_Module, ABC):  # type: ignore
    """Abstract interface hook for plug-and-play attention modules."""

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Apply attention modulation over feature map x."""
        pass


class ChannelAttention(_Module):  # type: ignore
    """Channel Attention Sub-module using MaxPool and AvgPool expectation features."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        reduced_channels = max(in_channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Any) -> Any:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(_Module):  # type: ignore
    """Spatial Attention Sub-module using channel-wise MaxPool and AvgPool maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Any) -> Any:
        if torch is None:
            return x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale_map = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale_map))
        return x * scale


class CBAM(AttentionHook):
    """Convolutional Block Attention Module combining channel and spatial attention."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attn = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: Any) -> Any:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


__all__ = ["AttentionHook", "ChannelAttention", "SpatialAttention", "CBAM"]
