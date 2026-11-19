"""Spatial & Channel Attention Modules for Aethel Detector Backbone.

Provides ChannelAttention (Squeeze-and-Excitation), SpatialAttention, and CBAM
(Convolutional Block Attention Module) to focus feature maps on salient regions.
"""

from typing import Optional

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention (Squeeze-and-Excitation) module for Aethel Detector.

    Squeezes spatial dimensions via AdaptiveAvgPool2d and AdaptiveMaxPool2d,
    passing through excitation layers to generate channel saliency weights.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize ChannelAttention.

        Args:
            channels: Number of feature channels.
            reduction: Channel squeeze reduction ratio integer (default 16).
        """
        super().__init__()
        mid_ch = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ChannelAttention.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            Channel-weighted feature tensor of shape (B, C, H, W).
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial Attention module for Aethel Detector.

    Computes spatial average and max pooling across channels, passing through
    a 7x7 convolution to generate a spatial attention mask.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize SpatialAttention.

        Args:
            kernel_size: Spatial convolution kernel size integer (default 7).
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SpatialAttention.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            Spatially-weighted feature tensor of shape (B, C, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(concat))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM) for Aethel Detector.

    Combines ChannelAttention and SpatialAttention sequentially.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        """Initialize CBAM module.

        Args:
            channels: Number of feature channels.
            reduction: Channel reduction ratio integer.
            spatial_kernel: Spatial attention convolution kernel size.
        """
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for CBAM.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            Attention-refined feature tensor of shape (B, C, H, W).
        """
        x = self.channel_att(x)
        return self.spatial_att(x)


__all__ = ["ChannelAttention", "SpatialAttention", "CBAM"]
