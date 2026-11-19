"""Composite Loss Orchestrator for Aethel Vision Framework.

Dynamically combines classification loss (FocalLoss), bounding box regression loss (CIoU),
and distribution focal loss (DFL) driven by configuration specs.
"""

from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore

from aethel.losses.base import BaseLoss
from aethel.losses.dfl import DistributionFocalLoss
from aethel.losses.focal import FocalLoss
from aethel.losses.iou import BboxIoULoss
from aethel.registry.registry import LOSS_REGISTRY
from aethel.utils.logger import get_logger

logger = get_logger("aethel.losses.composite")


@LOSS_REGISTRY.register("CompositeDetectionLoss")
class CompositeDetectionLoss(BaseLoss):
    """Dynamic composite loss combining classification, IoU box regression, and DFL losses."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CompositeDetectionLoss from configuration spec.

        Args:
            config: Loss configuration dictionary specifying weights for box, cls, and dfl.
        """
        super().__init__()
        self.config = config or {}

        # Loss weights
        self.box_weight = self.config.get("box_weight", 7.5)
        self.cls_weight = self.config.get("cls_weight", 0.5)
        self.dfl_weight = self.config.get("dfl_weight", 1.5)

        # Sub-loss modules
        iou_type = self.config.get("iou_type", "ciou")
        self.iou_loss = BboxIoULoss(loss_type=iou_type)
        self.cls_loss = FocalLoss(
            alpha=self.config.get("focal_alpha", 0.25),
            gamma=self.config.get("focal_gamma", 2.0),
        )
        self.dfl_loss = DistributionFocalLoss(reg_max=self.config.get("reg_max", 16))

    def forward(self, predictions: Any, targets: Any) -> Tuple[Any, Dict[str, float]]:
        """Compute weighted composite loss.

        Args:
            predictions: Model output or dict containing ('cls_logits', 'pred_boxes', 'pred_dfl').
            targets: Dict containing ('cls_targets', 'target_boxes', 'target_dfl').

        Returns:
            Tuple of (total_weighted_loss_tensor, breakdown_dict).
        """
        if torch is None or not isinstance(predictions, dict) or not isinstance(targets, dict):
            return 0.0, {"total_loss": 0.0}

        # 1. Compute classification loss
        cls_logits = predictions.get("cls_logits")
        cls_targets = targets.get("cls_targets")
        cls_val, cls_dict = (
            self.cls_loss(cls_logits, cls_targets)
            if cls_logits is not None and cls_targets is not None
            else (0.0, {})
        )

        # 2. Compute IoU box regression loss
        pred_boxes = predictions.get("pred_boxes")
        target_boxes = targets.get("target_boxes")
        box_val, box_dict = (
            self.iou_loss(pred_boxes, target_boxes)
            if pred_boxes is not None and target_boxes is not None
            else (0.0, {})
        )

        # 3. Compute DFL loss
        pred_dfl = predictions.get("pred_dfl")
        target_dfl = targets.get("target_dfl")
        dfl_val, dfl_dict = (
            self.dfl_loss(pred_dfl, target_dfl)
            if pred_dfl is not None and target_dfl is not None
            else (0.0, {})
        )

        # Total weighted loss
        total_loss = self.cls_weight * cls_val + self.box_weight * box_val + self.dfl_weight * dfl_val

        breakdown = {
            "total_loss": float(total_loss.item()) if hasattr(total_loss, "item") else float(total_loss),
            "cls_loss": float(cls_val.item()) if hasattr(cls_val, "item") else float(cls_val),
            "box_loss": float(box_val.item()) if hasattr(box_val, "item") else float(box_val),
            "dfl_loss": float(dfl_val.item()) if hasattr(dfl_val, "item") else float(dfl_val),
            "l_cls": float(cls_val.item()) if hasattr(cls_val, "item") else float(cls_val),
            "l_box": float(box_val.item()) if hasattr(box_val, "item") else float(box_val),
            "l_dfl": float(dfl_val.item()) if hasattr(dfl_val, "item") else float(dfl_val),
        }

        return total_loss, breakdown


# Aliases for interface compliance
DetectionLoss = CompositeDetectionLoss
ClassificationLoss = FocalLoss
BoundingBoxLoss = BboxIoULoss
LossComposer = CompositeDetectionLoss


__all__ = [
    "CompositeDetectionLoss",
    "DetectionLoss",
    "ClassificationLoss",
    "BoundingBoxLoss",
    "LossComposer",
]
