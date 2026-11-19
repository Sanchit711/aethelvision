"""Structural Block Primitives for Aethel Vision Framework.

Provides Bottleneck residual blocks, Cross Stage Partial (CSP) stages, and
Spatial Pyramid Pooling Fast (SPPF) modules.
"""

from typing import Any, List, Optional, Union

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
        MaxPool2d = _Dummy
    nn = _NN()  # type: ignore
    _Module = nn.Module

from aethel.layers.conv import Conv


class Bottleneck(_Module):  # type: ignore
    """Standard residual bottleneck block with optional shortcut connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        act: Union[str, bool, None] = True,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, act=act)
        self.cv2 = Conv(hidden_channels, out_channels, 3, 1, act=act)
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x: Any) -> Any:
        res = self.cv2(self.cv1(x))
        return x + res if self.add_shortcut else res


class CSPBlock(_Module):  # type: ignore
    """Cross Stage Partial (CSP) bottleneck block with multi-branch feature split."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        act: Union[str, bool, None] = True,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, act=act)
        self.cv2 = Conv(in_channels, hidden_channels, 1, 1, act=act)
        self.cv3 = Conv(2 * hidden_channels, out_channels, 1, 1, act=act)

        self.m = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut, expansion=1.0, act=act) for _ in range(num_blocks)]
        )

    def forward(self, x: Any) -> Any:
        if torch is None:
            return x
        branch1 = self.m(self.cv1(x))
        branch2 = self.cv2(x)
        return self.cv3(torch.cat([branch1, branch2], dim=1))


class SPPF(_Module):  # type: ignore
    """Spatial Pyramid Pooling - Fast (SPPF) layer for multi-receptive field feature fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        act: Union[str, bool, None] = True,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, act=act)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: Any) -> Any:
        if torch is None:
            return x
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


__all__ = ["Bottleneck", "CSPBlock", "SPPF"]
