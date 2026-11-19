"""Common Neural Network Layer Building Blocks for Aethel Detector.

Provides fundamental Conv2d wrappers, auto-padding calculations, and
Conv + BatchNorm + Activation (SiLU / ReLU) blocks.
"""

from typing import Any, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    class _Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any: return None
    nn = type("nn", (), {"Module": _Module})()  # type: ignore

try:
    from aethel_detector.layers.activation import get_activation
    from aethel_detector.layers.normalization import get_normalization
except Exception:
    get_activation = get_normalization = None  # type: ignore


def autopad(
    k: Union[int, Tuple[int, ...]],
    p: Optional[Union[int, Tuple[int, ...]]] = None,
    d: int = 1,
) -> Union[int, Tuple[int, ...]]:
    """Calculate same padding automatically based on kernel size and dilation."""
    if p is None:
        if isinstance(k, int):
            p = (k // 2) * d
        else:
            p = tuple((x // 2) * d for x in k)
    return p


class Conv(nn.Module):
    """Standard Convolution + Normalization + Activation block for Aethel Detector."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        k: Optional[int] = None,
        s: Optional[int] = None,
        p: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        act: Union[str, bool, Any, None] = True,
        norm: Union[str, bool, Any, None] = True,
    ) -> None:
        """Initialize Conv block."""
        super().__init__()
        ks = k if k is not None else kernel_size
        st = s if s is not None else stride
        pd = p if p is not None else padding

        pad_val = autopad(ks, pd, dilation)
        bias = norm is False or norm is None or norm == "identity"

        if torch is not None and hasattr(nn, "Conv2d"):
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ks,
                stride=st,
                padding=pad_val,
                groups=groups,
                dilation=dilation,
                bias=bias,
            )
            self.bn = get_normalization(norm, out_channels) if get_normalization is not None else nn.Identity()
            self.norm = self.bn
            self.act = get_activation(act) if get_activation is not None else nn.Identity()

    def forward(self, x: Any) -> Any:
        if not hasattr(self, "conv"):
            return x
        return self.act(self.bn(self.conv(x)))


class ConvBNReLU(Conv):
    """Conv + BatchNorm + ReLU layer block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act="relu",
            norm=True,
            **kwargs,
        )


class ConvBNSiLU(Conv):
    """Conv + BatchNorm + SiLU layer block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act="silu",
            norm=True,
            **kwargs,
        )


__all__ = ["autopad", "Conv", "ConvBNReLU", "ConvBNSiLU"]
