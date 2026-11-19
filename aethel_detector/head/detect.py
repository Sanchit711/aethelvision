"""Anchor-Free Decoupled Detection Head for Aethel Detector.

Predicts multi-scale object classification logits, bounding box distance regressions (DFL),
and objectness scores across N3 (stride 8), N4 (stride 16), and N5 (stride 32) neck feature levels.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from aethel_detector.layers.common import Conv


class DecoupledDetectHead(nn.Module):
    """Decoupled anchor-free multi-scale object detection head for Aethel Detector.

    For each scale (N3, N4, N5), splits feature paths into parallel:
    1. Classification Branch: Conv sequence -> (B, num_classes, H_i, W_i)
    2. Box Regression Branch: Conv sequence -> (B, 4 * reg_max, H_i, W_i)
    """

    def __init__(
        self,
        num_classes: int = 80,
        in_channels_list: Optional[Tuple[int, ...]] = None,
        reg_max: int = 16,
        act: Union[str, bool, nn.Module, None] = True,
        norm: Union[str, bool, nn.Module, None] = True,
        in_channels: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Initialize DecoupledDetectHead."""
        super().__init__()
        ch_list = in_channels_list if in_channels_list is not None else in_channels
        if ch_list is None:
            ch_list = (128, 256, 512)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_outputs = num_classes + 4 * reg_max
        self.strides = (8, 16, 32)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for in_ch in ch_list:
            # Classification branch (2 x Conv)
            self.cls_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.cls_preds.append(nn.Conv2d(in_ch, num_classes, kernel_size=1))

            # Bounding Box Regression branch (2 x Conv)
            self.reg_convs.append(
                nn.Sequential(
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                    Conv(in_ch, in_ch, kernel_size=3, stride=1, act=act, norm=norm),
                )
            )
            self.reg_preds.append(nn.Conv2d(in_ch, 4 * reg_max, kernel_size=1))

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> List[torch.Tensor]:
        """Forward pass predicting classification logits and box regression logits across scale levels.

        Args:
            features: Tuple of (N3, N4, N5) neck feature map tensors.

        Returns:
            List of raw prediction tensors for scale levels [N3_out, N4_out, N5_out].
            Each scale output has shape (B, num_classes + 4 * reg_max, H_i, W_i).
        """
        outputs = []
        for i, x in enumerate(features):
            cls_feat = self.cls_convs[i](x)
            cls_out = self.cls_preds[i](cls_feat)

            reg_feat = self.reg_convs[i](x)
            reg_out = self.reg_preds[i](reg_feat)

            out = torch.cat([cls_out, reg_out], dim=1)
            outputs.append(out)

        return outputs


__all__ = ["DecoupledDetectHead"]
