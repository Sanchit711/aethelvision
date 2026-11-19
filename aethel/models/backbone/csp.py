"""CSP Backbone Feature Extractor for Aethel Vision Framework.

Extracts multi-scale feature representations P3 (stride 8), P4 (stride 16), and P5 (stride 32)
using Cross Stage Partial stages and Spatial Pyramid Pooling Fast (SPPF).
"""

from typing import Any, List, Tuple, Union

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
        Identity = _Dummy
    nn = _NN()  # type: ignore
    _Module = nn.Module

from aethel.layers.attention import CBAM
from aethel.layers.blocks import SPPF, CSPBlock
from aethel.layers.conv import Conv
from aethel.models.backbone.stem import DetectorStem


def make_divisible(v: float, divisor: int = 8) -> int:
    """Ensure channel counts are divisible by divisor."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class CSPBackbone(_Module):  # type: ignore
    """CSP Backbone feature extractor producing multi-scale features (P3, P4, P5)."""

    def __init__(
        self,
        in_channels: int = 3,
        depth_multiple: float = 1.0,
        width_multiple: float = 1.0,
        act: Union[str, bool, None] = True,
        norm: bool = True,
        use_cbam: bool = True,
    ) -> None:
        """Initialize CSPBackbone."""
        super().__init__()
        c_stem = make_divisible(64 * width_multiple)
        c_p3 = make_divisible(128 * width_multiple)
        c_p4 = make_divisible(256 * width_multiple)
        c_p5 = make_divisible(512 * width_multiple)

        n_p3 = max(round(3 * depth_multiple), 1)
        n_p4 = max(round(6 * depth_multiple), 1)
        n_p5 = max(round(3 * depth_multiple), 1)

        self.stem = DetectorStem(in_channels=in_channels, stem_channels=c_stem, act=act, norm=norm)

        self.conv_p3 = Conv(c_stem, c_p3, kernel_size=3, stride=2, act=act, norm=norm)
        self.csp_p3 = CSPBlock(c_p3, c_p3, num_blocks=n_p3, shortcut=True, act=act)

        self.conv_p4 = Conv(c_p3, c_p4, kernel_size=3, stride=2, act=act, norm=norm)
        self.csp_p4 = CSPBlock(c_p4, c_p4, num_blocks=n_p4, shortcut=True, act=act)

        self.conv_p5 = Conv(c_p4, c_p5, kernel_size=3, stride=2, act=act, norm=norm)
        self.csp_p5 = CSPBlock(c_p5, c_p5, num_blocks=n_p5, shortcut=True, act=act)
        self.sppf = SPPF(c_p5, c_p5, kernel_size=5, act=act)
        self.cbam = CBAM(c_p5) if use_cbam else nn.Identity()

        self.out_channels = (c_p3, c_p4, c_p5)

    def forward(self, x: Any) -> Tuple[Any, Any, Any]:
        """Forward pass."""
        if torch is None:
            return (x, x, x)

        x_stem = self.stem(x)

        p3 = self.csp_p3(self.conv_p3(x_stem))
        p4 = self.csp_p4(self.conv_p4(p3))
        p5_feat = self.sppf(self.csp_p5(self.conv_p5(p4)))
        p5 = self.cbam(p5_feat)

        return (p3, p4, p5)


__all__ = ["CSPBackbone", "make_divisible"]
