"""Convolutional Feature Blocks for Aethel Detector Backbone.

Provides standard Conv, DepthwiseConv, ConvBNReLU, and ConvBNSiLU building blocks.
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from aethel_detector.layers.common import Conv, ConvBNReLU, ConvBNSiLU, autopad


class DepthwiseConv(Conv):
    """Depthwise Separable Convolution block (groups = in_channels)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        act: Union[str, bool, nn.Module, None] = True,
        norm: Union[str, bool, nn.Module, None] = True,
    ) -> None:
        """Initialize DepthwiseConv block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (must equal in_channels for pure depthwise).
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Optional explicit padding.
            dilation: Convolution dilation rate.
            act: Activation layer specification.
            norm: Normalization layer specification.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            act=act,
            norm=norm,
        )


__all__ = ["Conv", "DepthwiseConv", "ConvBNReLU", "ConvBNSiLU", "autopad"]
