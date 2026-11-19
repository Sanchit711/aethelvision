"""Aethel Indigenous Instance Segmentation Loss Module.

Features:
- MaskBCELoss: Pixel-wise binary cross-entropy loss evaluated on assembled instance masks.
- DiceLoss: Continuous soft Dice loss balancing sparse foreground segmentation masks.
- AethelSegmentLoss: Complete composite loss orchestrator combining bounding box,
  classification, DFL, prototype mask reconstruction, and instance mask losses.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from aethel.losses.base import BaseLoss
from aethel.losses.composite import CompositeDetectionLoss
from aethel.models.head.segment import assemble_masks
from aethel.registry.registry import LOSS_REGISTRY
from aethel.utils.logger import get_logger

logger = get_logger("aethel.losses.segment")


@LOSS_REGISTRY.register("DiceLoss")
class DiceLoss(BaseLoss):
    """Soft Dice Loss for Instance Segmentation Mask Optimization."""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred_masks: Any, target_masks: Any) -> Any:
        """Compute Soft Dice loss.

        Args:
            pred_masks: Predicted sigmoid mask probabilities (N, H, W).
            target_masks: Ground truth binary masks (N, H, W).

        Returns:
            Scalar Dice loss = 1.0 - mean(Dice score).
        """
        if torch is None or pred_masks is None or target_masks is None:
            return 0.0

        p_flat = pred_masks.contiguous().view(pred_masks.shape[0], -1)
        t_flat = target_masks.contiguous().view(target_masks.shape[0], -1).float()

        intersection = torch.sum(p_flat * t_flat, dim=1)
        cardinality = torch.sum(p_flat, dim=1) + torch.sum(t_flat, dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return torch.mean(1.0 - dice)


@LOSS_REGISTRY.register("MaskBCELoss")
class MaskBCELoss(BaseLoss):
    """Pixel-level Binary Cross-Entropy Loss for Instance Masks."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, pred_masks: Any, target_masks: Any) -> Any:
        """Compute BCE loss between predicted logits/probabilities and binary targets."""
        if torch is None or pred_masks is None or target_masks is None or F is None:
            return 0.0

        target_masks = target_masks.float()
        return F.binary_cross_entropy(pred_masks, target_masks, reduction="mean") * self.weight


@LOSS_REGISTRY.register("AethelSegmentLoss")
class AethelSegmentLoss(BaseLoss):
    """Comprehensive Loss Orchestrator for Real-Time Instance Segmentation.

    Combines:
    - Bounding Box CIoU Loss
    - Distribution Focal Loss (DFL)
    - Classification Focal Loss
    - Prototype-Assembled Mask Binary Cross-Entropy Loss
    - Soft Dice Segmentation Loss
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config = config or {}

        # Loss weights
        self.box_weight = self.config.get("box_weight", 7.5)
        self.cls_weight = self.config.get("cls_weight", 0.5)
        self.dfl_weight = self.config.get("dfl_weight", 1.5)
        self.mask_weight = self.config.get("mask_weight", 2.5)
        self.dice_weight = self.config.get("dice_weight", 1.0)

        # Base detection losses
        self.detection_loss = CompositeDetectionLoss(self.config)
        self.bce_loss = MaskBCELoss()
        self.dice_loss = DiceLoss()

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, float]]:
        """Compute composite instance segmentation loss.

        Args:
            predictions: Dict containing:
                - 'cls_logits': predicted class logits
                - 'pred_boxes': predicted bounding boxes (x1, y1, x2, y2)
                - 'pred_dfl': predicted DFL distribution tensor (optional)
                - 'protos': prototype mask maps (B, K, H/4, W/4)
                - 'mask_coeffs': mask coefficients per instance (N, K)
                - 'assembled_masks': pre-assembled masks (optional, N, H/4, W/4)
            targets: Dict containing:
                - 'cls_targets': target labels
                - 'target_boxes': target bounding boxes (x1, y1, x2, y2)
                - 'target_dfl': target DFL distribution (optional)
                - 'target_masks': target instance masks (N, H/4, W/4)

        Returns:
            Tuple of (total_loss_tensor, breakdown_dict).
        """
        if torch is None:
            return 0.0, {"loss_total": 0.0}

        # 1. Detection Base Loss (Box + Cls + DFL)
        det_loss, breakdown = self.detection_loss(predictions, targets)

        # 2. Mask Segmentation Loss
        target_masks = targets.get("target_masks")
        pred_masks = predictions.get("assembled_masks")

        # If assembled masks aren't precomputed, assemble from protos and coeffs
        if pred_masks is None and "protos" in predictions and "mask_coeffs" in predictions:
            protos = predictions["protos"]
            coeffs = predictions["mask_coeffs"]
            boxes = targets.get("target_boxes", predictions.get("pred_boxes"))
            if protos is not None and coeffs is not None and len(coeffs) > 0:
                pred_masks = assemble_masks(protos, coeffs, boxes)

        mask_loss_val = torch.tensor(0.0, device=det_loss.device if hasattr(det_loss, "device") else "cpu")
        dice_loss_val = torch.tensor(0.0, device=det_loss.device if hasattr(det_loss, "device") else "cpu")

        if pred_masks is not None and target_masks is not None and len(pred_masks) > 0:
            # Resize target masks if dimension mismatch
            if pred_masks.shape[-2:] != target_masks.shape[-2:] and F is not None:
                target_masks = F.interpolate(
                    target_masks.unsqueeze(1).float(),
                    size=pred_masks.shape[-2:],
                    mode="nearest",
                ).squeeze(1)

            mask_loss_val = self.bce_loss(pred_masks, target_masks)
            dice_loss_val = self.dice_loss(pred_masks, target_masks)

        total_loss = det_loss + self.mask_weight * mask_loss_val + self.dice_weight * dice_loss_val

        breakdown["loss_mask_bce"] = float(mask_loss_val.detach().item()) if hasattr(mask_loss_val, "item") else 0.0
        breakdown["loss_mask_dice"] = float(dice_loss_val.detach().item()) if hasattr(dice_loss_val, "item") else 0.0
        breakdown["loss_total"] = float(total_loss.detach().item()) if hasattr(total_loss, "item") else 0.0

        return total_loss, breakdown


__all__ = ["DiceLoss", "MaskBCELoss", "AethelSegmentLoss"]
