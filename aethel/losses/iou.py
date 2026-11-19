"""Bounding Box IoU Loss Functions for Aethel Vision Framework.

Provides IoU, GIoU, DIoU, and CIoU bounding box regression loss formulations.
"""

from typing import Any, Tuple

try:
    import torch
    import torch.nn as nn
    _Module = nn.Module
except Exception:
    torch = None  # type: ignore
    class _Module:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass

from aethel.losses.base import BaseLoss
from aethel.registry.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("BboxIoULoss")
class BboxIoULoss(BaseLoss):
    """Bounding Box IoU Loss (supports 'iou', 'giou', 'diou', 'ciou')."""

    def __init__(self, loss_type: str = "ciou", eps: float = 1e-7) -> None:
        """Initialize BboxIoULoss.

        Args:
            loss_type: Loss variant string ('iou', 'giou', 'diou', 'ciou').
            eps: Epsilon threshold preventing division by zero.
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.eps = eps

    def forward(
        self,
        pred_boxes: Any,
        target_boxes: Any,
    ) -> Tuple[Any, dict]:
        """Compute IoU loss between predicted and target bounding boxes [x1, y1, x2, y2]."""
        if torch is None or not hasattr(pred_boxes, "shape"):
            return 0.0, {"iou_loss": 0.0}

        if pred_boxes.shape[0] == 0:
            return pred_boxes.sum() * 0.0, {"iou_loss": 0.0}

        b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
        inter_area = inter_w * inter_h

        w1, h1 = (b1_x2 - b1_x1).clamp(min=0.0), (b1_y2 - b1_y1).clamp(min=0.0)
        w2, h2 = (b2_x2 - b2_x1).clamp(min=0.0), (b2_y2 - b2_y1).clamp(min=0.0)
        union_area = w1 * h1 + w2 * h2 - inter_area + self.eps

        iou = inter_area / union_area

        if self.loss_type == "iou":
            loss = 1.0 - iou
        else:
            cw = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp(min=0.0)
            ch = (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp(min=0.0)

            if self.loss_type == "giou":
                c_area = cw * ch + self.eps
                giou = iou - (c_area - union_area) / c_area
                loss = 1.0 - giou
            else:
                c2 = cw ** 2 + ch ** 2 + self.eps
                rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4.0
                diou = iou - rho2 / c2

                if self.loss_type == "diou":
                    loss = 1.0 - diou
                else:
                    v = (4.0 / (3.141592653589793 ** 2)) * torch.pow(torch.atan(w2 / (h2 + self.eps)) - torch.atan(w1 / (h1 + self.eps)), 2)
                    with torch.no_grad():
                        alpha = v / (1.0 - iou + v + self.eps)
                    ciou = diou - alpha * v
                    loss = 1.0 - ciou

        mean_loss = loss.mean()
        return mean_loss, {f"{self.loss_type}_loss": mean_loss.item()}


__all__ = ["BboxIoULoss"]
