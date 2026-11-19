"""Convolution Layer Primitives for Aethel Vision Framework.

Provides standard 2D convolutions (Conv) and depthwise separable convolutions (DWConv)
with configurable activation and normalization layers.
"""

from typing import Any, Optional, Union

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
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return args[0] if args else None
    class _NN:
        Module = _Dummy
        Conv2d = _Dummy
        BatchNorm2d = _Dummy
        Identity = _Dummy
        SiLU = _Dummy
        ReLU = _Dummy
        GELU = _Dummy
        Sequential = _Dummy
        ModuleList = _Dummy
        Upsample = _Dummy
    nn = _NN()  # type: ignore
    _Module = nn.Module


def get_activation(act: Union[str, bool, None]) -> Any:
    """Activation layer factory."""
    if act is True or act == "silu":
        return nn.SiLU(inplace=True)
    elif act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "gelu":
        return nn.GELU()
    elif isinstance(act, nn.Module):
        return act
    return nn.Identity()


class Conv(_Module):  # type: ignore
    """Standard 2D Convolution block: Conv2d + BatchNorm2d + Activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act: Union[str, bool, None] = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not norm,
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = get_activation(act)

    def forward(self, x: Any) -> Any:
        """Forward pass."""
        return self.act(self.norm(self.conv(x)))


class DWConv(Conv):
    """Depthwise Separable Convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        act: Union[str, bool, None] = True,
    ) -> None:
        import math
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=math.gcd(in_channels, out_channels),
            act=act,
        )


__all__ = ["Conv", "DWConv", "get_activation"]
