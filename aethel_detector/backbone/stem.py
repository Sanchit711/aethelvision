"""Input Stem Architecture for Aethel Detector Backbone.

Processes raw input image tensors (3 channels) into multi-channel feature representations
while performing efficient initial spatial downsampling (4x total downsampling).
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from aethel_detector.layers.common import Conv


class DetectorStem(nn.Module):
    """Input Stem block for Aethel Detector backbone.

    Applies sequential 3x3 strided convolutions (stride 2) to perform 4x downsampling
    from input images (B, C_in, H, W) to feature representations (B, C_out, H/4, W/4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        stem_channels: Optional[int] = None,
        act: Union[str, bool, nn.Module, None] = True,
        norm: Union[str, bool, nn.Module, None] = True,
    ) -> None:
        """Initialize DetectorStem.

        Args:
            in_channels: Input image channel count (default 3 for RGB).
            out_channels: Final output feature channel count (default 64).
            stem_channels: Intermediate stem channel count (defaults to out_channels // 2).
            act: Activation layer specification (default True / SiLU).
            norm: Normalization layer specification (default True / BatchNorm2d).
        """
        super().__init__()
        mid_ch = stem_channels or max(out_channels // 2, 16)

        # Stage 1: 3x3 Conv (stride 2 -> 2x downsampling)
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=mid_ch,
            kernel_size=3,
            stride=2,
            act=act,
            norm=norm,
        )
        # Stage 2: 3x3 Conv (stride 2 -> 4x total downsampling)
        self.conv2 = Conv(
            in_channels=mid_ch,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            act=act,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass executing 4x downsampling stem.

        Args:
            x: Input image tensor of shape (B, C_in, H, W).

        Returns:
            Output feature tensor of shape (B, out_channels, H/4, W/4).
        """
        return self.conv2(self.conv1(x))


__all__ = ["DetectorStem"]
