"""Aethel Indigenous Oriented Bounding Box (OBB) Loss Module.

Features:
- ProbIoULoss: Differentiable 2D Gaussian Bhattacharyya distance representation of rotated boxes.
  Completely eliminates angle boundary jump singularities and backpropagation discontinuities.
- ContinuousAngleLoss: Trigonometric cosine similarity loss on (cos 2θ, sin 2θ) representations.
- AethelOBBLoss: Complete composite loss orchestrator combining bounding box, classification,
  DFL, angle regression, and ProbIoU losses.
"""

import math
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
from aethel.models.head.obb import decode_angle
from aethel.registry.registry import LOSS_REGISTRY
from aethel.utils.logger import get_logger

logger = get_logger("aethel.losses.obb")


@LOSS_REGISTRY.register("ContinuousAngleLoss")
class ContinuousAngleLoss(BaseLoss):
    """Trigonometric Loss on continuous angle encoding (cos 2θ, sin 2θ)."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_trig: Any, target_angle: Any) -> Any:
        """Compute angle loss.

        Args:
            pred_trig: Predicted (cos 2θ, sin 2θ) tensor (N, 2).
            target_angle: Target ground-truth angles in radians (N, 1) or (N,).

        Returns:
            Scalar angle loss in range [0, 2].
        """
        if torch is None or pred_trig is None or target_angle is None:
            return 0.0

        if target_angle.dim() == 1:
            target_angle = target_angle.unsqueeze(-1)

        # Normalize predicted (cos 2θ, sin 2θ)
        norm = torch.sqrt(pred_trig[:, 0:1] ** 2 + pred_trig[:, 1:2] ** 2 + self.eps)
        p_cos = pred_trig[:, 0:1] / norm
        p_sin = pred_trig[:, 1:2] / norm

        # Target (cos 2θ, sin 2θ)
        t_cos = torch.cos(2.0 * target_angle)
        t_sin = torch.sin(2.0 * target_angle)

        # Cosine difference: 1.0 - (p_cos * t_cos + p_sin * t_sin)
        cos_sim = p_cos * t_cos + p_sin * t_sin
        loss = 1.0 - cos_sim
        return torch.mean(loss)


@LOSS_REGISTRY.register("ProbIoULoss")
class ProbIoULoss(BaseLoss):
    """Probabilistic IoU Loss for Oriented Bounding Boxes via 2D Gaussian Modeling.

    Each rotated box (cx, cy, w, h, θ) is modeled as a 2D Gaussian distribution N(μ, Σ).
    The similarity is measured via the Bhattacharyya distance, giving smooth gradients
    everywhere without polygon clipping artifacts.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def _box_to_gaussian(self, boxes: Any) -> Tuple[Any, Any, Any, Any]:
        """Convert rotated boxes (cx, cy, w, h, θ) into 2D Gaussian parameters.

        Returns:
            μ_x, μ_y, Σ_11, Σ_22, Σ_12
        """
        x, y, w, h, theta = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3], boxes[..., 4]
        a = (w ** 2) / 12.0
        b = (h ** 2) / 12.0
        c = torch.cos(theta)
        s = torch.sin(theta)

        sigma_11 = a * (c ** 2) + b * (s ** 2)
        sigma_22 = a * (s ** 2) + b * (c ** 2)
        sigma_12 = (a - b) * c * s

        return (x, y), (sigma_11, sigma_22, sigma_12)

    def forward(self, pred_obb: Any, target_obb: Any) -> Any:
        """Compute ProbIoU loss between predicted and target rotated boxes.

        Args:
            pred_obb: Predicted boxes (N, 5) with (cx, cy, w, h, θ).
            target_obb: Target boxes (N, 5) with (cx, cy, w, h, θ).

        Returns:
            Scalar ProbIoU loss = 1.0 - ProbIoU.
        """
        if torch is None or pred_obb is None or target_obb is None:
            return 0.0

        (p_x, p_y), (p_s11, p_s22, p_s12) = self._box_to_gaussian(pred_obb)
        (t_x, t_y), (t_s11, t_s22, t_s12) = self._box_to_gaussian(target_obb)

        # Average covariance
        s11 = 0.5 * (p_s11 + t_s11)
        s22 = 0.5 * (p_s22 + t_s22)
        s12 = 0.5 * (p_s12 + t_s12)

        # Determinant of average covariance
        det_s = s11 * s22 - s12 ** 2 + self.eps

        # Inverse of average covariance components
        inv_s11 = s22 / det_s
        inv_s22 = s11 / det_s
        inv_s12 = -s12 / det_s

        # Position difference
        dx = p_x - t_x
        dy = p_y - t_y

        # Bhattacharyya distance term 1: (1/8) * Δμ^T * Σ^-1 * Δμ
        b1 = 0.125 * (dx * (dx * inv_s11 + dy * inv_s12) + dy * (dx * inv_s12 + dy * inv_s22))

        # Determinants of individual covariances
        det_p = p_s11 * p_s22 - p_s12 ** 2 + self.eps
        det_t = t_s11 * t_s22 - t_s12 ** 2 + self.eps

        # Bhattacharyya distance term 2: (1/2) * ln(det(Σ) / sqrt(det(Σ_1) * det(Σ_2)))
        b2 = 0.5 * torch.log(det_s / (torch.sqrt(det_p * det_t) + self.eps) + self.eps)

        bd = torch.clamp(b1 + b2, min=0.0, max=100.0)

        # ProbIoU = e^(-bd)
        probiou = torch.exp(-bd)
        loss = 1.0 - probiou
        return torch.mean(loss)


@LOSS_REGISTRY.register("AethelOBBLoss")
class AethelOBBLoss(BaseLoss):
    """Comprehensive Loss Orchestrator for Oriented Bounding Box Object Detection.

    Combines:
    - Bounding Box IoU / CIoU Loss
    - Distribution Focal Loss (DFL)
    - Classification Focal Loss
    - Continuous Trigonometric Angle Loss
    - Gaussian ProbIoU Loss
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config = config or {}

        # Loss weights
        self.box_weight = self.config.get("box_weight", 7.5)
        self.cls_weight = self.config.get("cls_weight", 0.5)
        self.dfl_weight = self.config.get("dfl_weight", 1.5)
        self.angle_weight = self.config.get("angle_weight", 2.0)
        self.probiou_weight = self.config.get("probiou_weight", 2.5)

        # Base detection losses
        self.detection_loss = CompositeDetectionLoss(self.config)
        self.angle_loss = ContinuousAngleLoss()
        self.probiou_loss = ProbIoULoss()

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, float]]:
        """Compute composite OBB loss.

        Args:
            predictions: Dict containing:
                - 'cls_logits': predicted class logits
                - 'pred_boxes': predicted bounding boxes (x1, y1, x2, y2)
                - 'pred_dfl': predicted DFL distribution tensor (optional)
                - 'pred_angle_trig': (cos 2θ, sin 2θ) predictions (N, 2)
                - 'pred_obb': complete (cx, cy, w, h, θ) predictions (optional, N, 5)
            targets: Dict containing:
                - 'cls_targets': target labels
                - 'target_boxes': target bounding boxes (x1, y1, x2, y2)
                - 'target_dfl': target DFL distribution (optional)
                - 'target_angles': target angles in radians (N, 1) or (N,)
                - 'target_obb': target rotated boxes (N, 5)

        Returns:
            Tuple of (total_loss_tensor, breakdown_dict).
        """
        if torch is None:
            return 0.0, {"loss_total": 0.0}

        # 1. Detection Base Loss (Box + Cls + DFL)
        det_loss, breakdown = self.detection_loss(predictions, targets)

        # 2. Angle Loss
        pred_trig = predictions.get("pred_angle_trig")
        target_angles = targets.get("target_angles")

        angle_loss_val = torch.tensor(0.0, device=det_loss.device if hasattr(det_loss, "device") else "cpu")
        if pred_trig is not None and target_angles is not None:
            angle_loss_val = self.angle_loss(pred_trig, target_angles)

        # 3. ProbIoU Loss
        pred_obb = predictions.get("pred_obb")
        target_obb = targets.get("target_obb")

        # Assemble pred_obb if not precomputed
        if pred_obb is None and "pred_boxes" in predictions and pred_trig is not None:
            boxes = predictions["pred_boxes"]
            if len(boxes) > 0:
                cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
                cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
                w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-3)
                h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-3)
                theta = decode_angle(pred_trig)
                pred_obb = torch.stack([cx, cy, w, h, theta], dim=-1)

        probiou_loss_val = torch.tensor(0.0, device=det_loss.device if hasattr(det_loss, "device") else "cpu")
        if pred_obb is not None and target_obb is not None and len(pred_obb) > 0:
            probiou_loss_val = self.probiou_loss(pred_obb, target_obb)

        total_loss = (
            det_loss
            + self.angle_weight * angle_loss_val
            + self.probiou_weight * probiou_loss_val
        )

        breakdown["loss_angle"] = float(angle_loss_val.detach().item()) if hasattr(angle_loss_val, "item") else 0.0
        breakdown["loss_probiou"] = float(probiou_loss_val.detach().item()) if hasattr(probiou_loss_val, "item") else 0.0
        breakdown["loss_total"] = float(total_loss.detach().item()) if hasattr(total_loss, "item") else 0.0

        return total_loss, breakdown


__all__ = ["ContinuousAngleLoss", "ProbIoULoss", "AethelOBBLoss"]
