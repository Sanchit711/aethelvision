"""Path Aggregation Feature Pyramid Network (PAFPN) for Aethel Detector Neck.

Combines top-down FPN feature propagation with bottom-up PANet path aggregation.
"""

from typing import Any, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    class _Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any: return []
    nn = type("nn", (), {"Module": _Module})()  # type: ignore

try:
    from aethel_detector.backbone.csp import CSPStage
    from aethel_detector.layers.common import Conv
    from aethel_detector.neck.fpn import FeaturePyramidNetwork
    from aethel_detector.neck.fusion import FeatureFusionBlock
except Exception:
    CSPStage = Conv = FeaturePyramidNetwork = FeatureFusionBlock = None  # type: ignore


class PAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network (PAFPN) module."""

    def __init__(
        self,
        in_channels_list: Optional[Tuple[int, int, int]] = None,
        in_channels: Optional[Tuple[int, int, int]] = None,
        out_channels: Optional[Tuple[int, int, int]] = None,
        act: Union[str, bool, Any, None] = True,
        norm: Union[str, bool, Any, None] = True,
    ) -> None:
        """Initialize PAFPN.

        Args:
            in_channels_list: Tuple of channel counts for (P3, P4, P5).
            in_channels: Alias for in_channels_list.
            out_channels: Output channels tuple.
            act: Activation layer specification.
            norm: Normalization layer specification.
        """
        super().__init__()
        ch_tuple = in_channels_list or in_channels or (128, 256, 512)
        c3, c4, c5 = ch_tuple
        self.in_channels_list = ch_tuple

        if FeaturePyramidNetwork is not None:
            self.fpn = FeaturePyramidNetwork(in_channels_list=ch_tuple, act=act, norm=norm)
            self.fusion_p4_p5 = FeatureFusionBlock(in_channels_high=c5, in_channels_low=c4, out_channels=c4)
            self.fusion_p3_p4 = FeatureFusionBlock(in_channels_high=c4, in_channels_low=c3, out_channels=c3)

            self.bu_conv1 = Conv(c3, c3, k=3, s=2, p=1, act=act, norm=norm)
            self.bu_csp1 = CSPStage(c3 + c4, c4, num_blocks=1, act=act, norm=norm)
            self.bu_conv2 = Conv(c4, c4, k=3, s=2, p=1, act=act, norm=norm)
            self.bu_csp2 = CSPStage(c4 + c5, c5, num_blocks=1, act=act, norm=norm)

    def forward(self, feats: Tuple[Any, Any, Any]) -> Tuple[Any, Any, Any]:
        """Execute PAFPN forward pass.

        Args:
            feats: Tuple of backbone feature tensors (P3, P4, P5).

        Returns:
            Tuple of PAFPN feature tensors (N3, N4, N5).
        """
        if not hasattr(self, "fpn"):
            return feats

        p3_fpn, p4_fpn, p5_fpn = self.fpn(feats)

        n3 = p3_fpn
        p3_down = self.bu_conv1(n3)
        p4_cat = torch.cat([p3_down, p4_fpn], dim=1) if torch is not None else p4_fpn
        n4 = self.bu_csp1(p4_cat)

        p4_down = self.bu_conv2(n4)
        p5_cat = torch.cat([p4_down, p5_fpn], dim=1) if torch is not None else p5_fpn
        n5 = self.bu_csp2(p5_cat)

        return (n3, n4, n5)


__all__ = ["PAFPN"]
