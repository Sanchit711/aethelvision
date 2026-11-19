"""Cross Stage Partial (CSP) Modules for Aethel Detector Backbone.

Provides CSPStage feature processing blocks that split feature maps into
parallel processing paths to optimize memory throughput and gradient combination.
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from aethel_detector.layers.common import Conv
from aethel_detector.backbone.residual import ResidualBlock


class CSPStage(nn.Module):
    """Cross Stage Partial (CSP) Bottleneck Stage block for Aethel Detector.

    Splits input channels into two parallel processing paths:
    Path 1: Passes through 1x1 conv -> sequence of N ResidualBlocks -> 1x1 conv.
    Path 2: Passes through 1x1 conv in parallel.
    Paths are concatenated and fused through a final 1x1 convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        act: Union[str, bool, nn.Module, None] = True,
        norm: Union[str, bool, nn.Module, None] = True,
    ) -> None:
        """Initialize CSPStage.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_blocks: Number of sequential ResidualBlocks in path 1 (default 1).
            shortcut: If True, enable residual shortcut connections inside ResidualBlocks.
            expansion: Hidden channel expansion ratio float (default 0.5).
            act: Activation layer specification (default True / SiLU).
            norm: Normalization layer specification (default True / BatchNorm2d).
        """
        super().__init__()
        hidden_ch = int(out_channels * expansion)

        # Path 1: Conv -> ResidualBlocks -> Conv
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_ch,
            kernel_size=1,
            stride=1,
            act=act,
            norm=norm,
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    in_channels=hidden_ch,
                    out_channels=hidden_ch,
                    shortcut=shortcut,
                    expansion=1.0,
                    act=act,
                    norm=norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv2 = Conv(
            in_channels=hidden_ch,
            out_channels=hidden_ch,
            kernel_size=1,
            stride=1,
            act=act,
            norm=norm,
        )

        # Path 2: Parallel 1x1 Conv
        self.conv3 = Conv(
            in_channels=in_channels,
            out_channels=hidden_ch,
            kernel_size=1,
            stride=1,
            act=act,
            norm=norm,
        )

        # Final fusion Conv: Concatenated (hidden_ch + hidden_ch) -> out_channels
        self.conv_fusion = Conv(
            in_channels=hidden_ch * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            act=act,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for CSPStage.

        Args:
            x: Input feature tensor of shape (B, in_channels, H, W).

        Returns:
            Output feature tensor of shape (B, out_channels, H, W).
        """
        path1 = self.conv2(self.blocks(self.conv1(x)))
        path2 = self.conv3(x)
        return self.conv_fusion(torch.cat((path1, path2), dim=1))


__all__ = ["CSPStage"]
