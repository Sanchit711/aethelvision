"""Top-Down Feature Pyramid Network (FPN) for Aethel Detector Neck.

Propagates high-level semantic context top-down from P5 to P4 and P3.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn

from aethel_detector.backbone.csp import CSPStage
from aethel_detector.layers.common import Conv
from aethel_detector.neck.fusion import FeatureFusionBlock, UpsampleBlock


class FeaturePyramidNetwork(nn.Module):
    """Top-Down Feature Pyramid Network (FPN) module.

    Propagates deep semantic information from P5 down to P4 and P3 feature maps.
    """

    def __init__(
        self,
        in_channels_list: Tuple[int, int, int] = (128, 256, 512),
        act: Union[str, bool, nn.Module, None] = True,
        norm: Union[str, bool, nn.Module, None] = True,
    ) -> None:
        """Initialize FeaturePyramidNetwork.

        Args:
            in_channels_list: Tuple of input channels for (P3, P4, P5).
            act: Activation layer specification.
            norm: Normalization layer specification.
        """
        super().__init__()
        c3, c4, c5 = in_channels_list

        self.p5_lateral = Conv(c5, c5, kernel_size=1, stride=1, act=act, norm=norm)
        self.upsample5_to_4 = UpsampleBlock(scale_factor=2)
        self.p4_fusion = FeatureFusionBlock(in_channels=c5 + c4, out_channels=c4, act=act, norm=norm)
        self.p4_csp = CSPStage(c4, c4, num_blocks=1, act=act, norm=norm)

        self.upsample4_to_3 = UpsampleBlock(scale_factor=2)
        self.p3_fusion = FeatureFusionBlock(in_channels=c4 + c3, out_channels=c3, act=act, norm=norm)
        self.p3_csp = CSPStage(c3, c3, num_blocks=1, act=act, norm=norm)

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass propagating top-down semantic context.

        Args:
            features: Tuple of (P3, P4, P5) feature map tensors from backbone.

        Returns:
            Tuple of top-down fused feature maps (F3, F4, F5).
        """
        p3, p4, p5 = features

        f5 = self.p5_lateral(p5)

        f5_upsampled = self.upsample5_to_4(f5)
        f4 = self.p4_csp(self.p4_fusion(f5_upsampled, p4))

        f4_upsampled = self.upsample4_to_3(f4)
        f3 = self.p3_csp(self.p3_fusion(f4_upsampled, p3))

        return f3, f4, f5


__all__ = ["FeaturePyramidNetwork"]
