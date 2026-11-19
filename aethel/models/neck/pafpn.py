"""Path Aggregation Feature Pyramid Network (PAFPN) Neck for Aethel Vision.

Fuses multi-scale backbone feature maps P3, P4, P5 via top-down feature enhancement
and bottom-up path aggregation to produce feature maps N3, N4, N5.
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
        Upsample = _Dummy
    nn = _NN()  # type: ignore
    _Module = nn.Module

from aethel.layers.blocks import CSPBlock
from aethel.layers.conv import Conv


class PAFPN(_Module):  # type: ignore
    """Path Aggregation Feature Pyramid Network neck module."""

    def __init__(
        self,
        in_channels_list: Tuple[int, int, int] = (128, 256, 512),
        depth_multiple: float = 1.0,
        act: Union[str, bool, None] = True,
        norm: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize PAFPN."""
        super().__init__()
        c_p3, c_p4, c_p5 = in_channels_list
        num_blocks = max(round(3 * depth_multiple), 1)

        # Top-Down Pathway
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.reduce_p5 = Conv(c_p5, c_p4, kernel_size=1, stride=1, act=act, norm=norm)
        self.csp_top_p4 = CSPBlock(c_p4 + c_p4, c_p4, num_blocks=num_blocks, shortcut=False, act=act)

        self.reduce_p4 = Conv(c_p4, c_p3, kernel_size=1, stride=1, act=act, norm=norm)
        self.csp_top_p3 = CSPBlock(c_p3 + c_p3, c_p3, num_blocks=num_blocks, shortcut=False, act=act)

        # Bottom-Up Pathway
        self.down_n3 = Conv(c_p3, c_p3, kernel_size=3, stride=2, act=act, norm=norm)
        self.csp_bot_n4 = CSPBlock(c_p3 + c_p4, c_p4, num_blocks=num_blocks, shortcut=False, act=act)

        self.down_n4 = Conv(c_p4, c_p4, kernel_size=3, stride=2, act=act, norm=norm)
        self.csp_bot_n5 = CSPBlock(c_p4 + c_p5, c_p5, num_blocks=num_blocks, shortcut=False, act=act)

        self.out_channels = (c_p3, c_p4, c_p5)

    def forward(
        self,
        features: Tuple[Any, Any, Any],
    ) -> Tuple[Any, Any, Any]:
        """Forward pass."""
        if torch is None:
            return features

        p3, p4, p5 = features

        p5_reduced = self.reduce_p5(p5)
        p4_fused = self.csp_top_p4(torch.cat([self.upsample(p5_reduced), p4], dim=1))

        p4_reduced = self.reduce_p4(p4_fused)
        n3 = self.csp_top_p3(torch.cat([self.upsample(p4_reduced), p3], dim=1))

        n3_down = self.down_n3(n3)
        n4 = self.csp_bot_n4(torch.cat([n3_down, p4_fused], dim=1))

        n4_down = self.down_n4(n4)
        n5 = self.csp_bot_n5(torch.cat([n4_down, p5], dim=1))

        return (n3, n4, n5)


__all__ = ["PAFPN"]
