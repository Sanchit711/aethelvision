"""Bounding Box Intersection over Union (IoU, GIoU, CIoU) Losses for Aethel Detector.

Provides pure PyTorch numerical loss modules for bounding box spatial regression.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    iou_type: str = "ciou",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate pairwise or element-wise IoU (IoU / GIoU / CIoU) between boxes.

    Args:
        box1: Bounding boxes tensor of shape (N, 4) with [x1, y1, x2, y2].
        box2: Bounding boxes tensor of shape (N, 4) with [x1, y1, x2, y2].
        iou_type: IoU metric type ('ciou', 'giou', 'iou').
        eps: Epsilon for numerical stability.

    Returns:
        1D Tensor of shape (N,) containing computed IoU values.
    """
    if hasattr(box1, "device") and hasattr(box2, "device") and box1.device != box2.device:
        box2 = box2.to(box1.device)

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection coordinates
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Individual box areas
    w1, h1 = (b1_x2 - b1_x1).clamp(min=0), (b1_y2 - b1_y1).clamp(min=0)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=0), (b2_y2 - b2_y1).clamp(min=0)

    union_area = w1 * h1 + w2 * h2 - inter_area + eps
    iou = inter_area / union_area

    if iou_type.lower() == "iou":
        return iou

    # Enclosing box dimensions
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    if iou_type.lower() == "giou":
        c_area = cw * ch + eps
        return iou - (c_area - union_area) / c_area

    # Center distance squared
    c2 = cw**2 + ch**2 + eps
    rho2 = ((b1_x1 + b1_x2) - (b2_x1 + b2_x2)) ** 2 / 4 + ((b1_y1 + b1_y2) - (b2_y1 + b2_y2)) ** 2 / 4

    # CIoU aspect ratio penalty
    v = (4 / (math.pi**2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - (rho2 / c2 + v * alpha)


class IoULoss(nn.Module):
    """Standard IoU Loss module."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 1 - IoU loss."""
        if pred.numel() == 0:
            return pred.sum() * 0.0
        return (1.0 - bbox_iou(pred, target, iou_type="iou")).mean()


class GIoULoss(nn.Module):
    """Generalized IoU (GIoU) Loss module."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 1 - GIoU loss."""
        if pred.numel() == 0:
            return pred.sum() * 0.0
        return (1.0 - bbox_iou(pred, target, iou_type="giou")).mean()


class CIoULoss(nn.Module):
    """Complete IoU (CIoU) Loss module."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 1 - CIoU loss."""
        if pred.numel() == 0:
            return pred.sum() * 0.0
        return (1.0 - bbox_iou(pred, target, iou_type="ciou")).mean()


class BboxIoULoss(nn.Module):
    """Bounding Box IoU Loss wrapper."""

    def __init__(self, iou_type: str = "ciou") -> None:
        super().__init__()
        self.iou_type = iou_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.sum() * 0.0
        return (1.0 - bbox_iou(pred, target, iou_type=self.iou_type)).mean()


__all__ = ["bbox_iou", "IoULoss", "GIoULoss", "CIoULoss", "BboxIoULoss"]
