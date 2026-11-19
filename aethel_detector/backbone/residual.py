"""Residual Bottleneck Modules for Aethel Detector Backbone.

Provides residual bottleneck blocks and shortcut connections to prevent gradient
degradation across deep backbone feature extraction stages.
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from aethel_detector.layers.common import Conv


class ResidualBlock(nn.Module):
    """Standard Residual Bottleneck block for Aethel Detector.

    Applies a 1x1 squeeze convolution followed by a 3x3 expansion convolution,
    adding an identity shortcut connection if shortcut=True and in_channels == out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        act: Union[str, bool, nn.Module, None] = True,
        norm: Union[str, bool, nn.Module, None] = True,
    ) -> None:
        """Initialize ResidualBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            shortcut: If True, add identity shortcut connection when in_channels == out_channels.
            expansion: Channel squeeze factor float (default 0.5).
            act: Activation layer specification (default True / SiLU).
            norm: Normalization layer specification (default True / BatchNorm2d).
        """
        super().__init__()
        hidden_ch = int(out_channels * expansion)
        self.add_shortcut = shortcut and (in_channels == out_channels)

        # Conv 1: 1x1 channel squeeze/reduction
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_ch,
            kernel_size=1,
            stride=1,
            act=act,
            norm=norm,
        )
        # Conv 2: 3x3 channel expansion
        self.conv2 = Conv(
            in_channels=hidden_ch,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            act=act,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ResidualBlock.

        Args:
            x: Input feature tensor of shape (B, in_channels, H, W).

        Returns:
            Output feature tensor of shape (B, out_channels, H, W).
        """
        out = self.conv2(self.conv1(x))
        return x + out if self.add_shortcut else out


__all__ = ["ResidualBlock"]
