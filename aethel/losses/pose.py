"""Aethel Indigenous Human Pose Estimation Loss Module.

Features:
- OKSLoss: Object Keypoint Similarity loss formulated for 17 standard COCO joints.
- KeypointL1Loss: Smooth L1 / L1 regression masked by ground truth visibility flags.
- AethelPoseLoss: Complete composite loss orchestrator combining bounding box,
  classification, DFL, keypoint localization, and keypoint visibility losses.
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
from aethel.registry.registry import LOSS_REGISTRY
from aethel.utils.logger import get_logger

logger = get_logger("aethel.losses.pose")

# COCO Keypoint per-joint standard variance constants (sigmas)
# Order: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
#        left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
#        left_knee, right_knee, left_ankle, right_ankle
COCO_SIGMAS = [
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
]


@LOSS_REGISTRY.register("OKSLoss")
class OKSLoss(BaseLoss):
    """Object Keypoint Similarity (OKS) Loss for keypoint coordinate optimization."""

    def __init__(self, sigmas: Optional[List[float]] = None, eps: float = 1e-7) -> None:
        super().__init__()
        self.sigmas_list = sigmas or COCO_SIGMAS
        self.eps = eps
        if torch is not None:
            sigmas_t = torch.tensor(self.sigmas_list, dtype=torch.float32)
            # 2 * sigma^2
            self.register_buffer("two_sigmas_sq", 2.0 * (sigmas_t ** 2))
        else:
            self.two_sigmas_sq = None

    def forward(
        self,
        pred_kpts: Any,
        target_kpts: Any,
        target_areas: Any,
        visibility_mask: Optional[Any] = None,
    ) -> Any:
        """Compute OKS loss.

        Args:
            pred_kpts: Predicted coordinates (N, K, 2).
            target_kpts: Ground-truth coordinates (N, K, 2).
            target_areas: Object bounding box area (N, 1) or (N,).
            visibility_mask: Ground-truth visibility (N, K) with values > 0 for visible.

        Returns:
            Scalar OKS loss = 1.0 - mean(OKS).
        """
        if torch is None or pred_kpts is None:
            return 0.0

        if target_areas.dim() == 1:
            target_areas = target_areas.unsqueeze(-1)  # (N, 1)

        # Squared Euclidean distance per keypoint: (N, K)
        d_sq = torch.sum((pred_kpts - target_kpts) ** 2, dim=-1)

        # Normalization factor per keypoint: 2 * sigma_i^2 * area
        # two_sigmas_sq: (K,) -> (1, K)
        sigmas_factor = self.two_sigmas_sq.to(pred_kpts.device).unsqueeze(0)
        norm_factor = target_areas * sigmas_factor + self.eps

        # Per-keypoint similarity e^(-d^2 / (2 * s * sigma^2))
        e = torch.exp(-d_sq / norm_factor)

        # Visibility weighting
        if visibility_mask is not None:
            mask = (visibility_mask > 0).float()
            oks_per_instance = torch.sum(e * mask, dim=-1) / (torch.sum(mask, dim=-1) + self.eps)
        else:
            oks_per_instance = torch.mean(e, dim=-1)

        loss = 1.0 - oks_per_instance
        return torch.mean(loss)


@LOSS_REGISTRY.register("KeypointL1Loss")
class KeypointL1Loss(BaseLoss):
    """Visibility-Masked Smooth L1 Loss for Keypoint Coordinates."""

    def __init__(self, beta: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        pred_kpts: Any,
        target_kpts: Any,
        visibility_mask: Optional[Any] = None,
    ) -> Any:
        """Compute Smooth L1 loss masked by visibility."""
        if torch is None or pred_kpts is None:
            return 0.0

        diff = torch.abs(pred_kpts - target_kpts)
        if self.beta > 0:
            smooth_l1 = torch.where(diff < self.beta, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta)
        else:
            smooth_l1 = diff

        l1_per_kpt = torch.sum(smooth_l1, dim=-1)  # (N, K)

        if visibility_mask is not None:
            mask = (visibility_mask > 0).float()
            loss = torch.sum(l1_per_kpt * mask) / (torch.sum(mask) + self.eps)
        else:
            loss = torch.mean(l1_per_kpt)

        return loss


@LOSS_REGISTRY.register("AethelPoseLoss")
class AethelPoseLoss(BaseLoss):
    """Comprehensive Loss Orchestrator for Real-Time Human Pose Estimation.

    Combines:
    - Bounding Box CIoU Loss
    - Distribution Focal Loss (DFL)
    - Classification Focal Loss
    - Object Keypoint Similarity (OKS) Loss
    - Keypoint Visibility BCE Loss
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config = config or {}

        # Loss weights
        self.box_weight = self.config.get("box_weight", 7.5)
        self.cls_weight = self.config.get("cls_weight", 0.5)
        self.dfl_weight = self.config.get("dfl_weight", 1.5)
        self.kpt_weight = self.config.get("kpt_weight", 12.0)
        self.kpt_vis_weight = self.config.get("kpt_vis_weight", 1.0)

        # Base detection losses
        self.detection_loss = CompositeDetectionLoss(self.config)
        self.oks_loss = OKSLoss()
        self.kpt_l1_loss = KeypointL1Loss()

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, float]]:
        """Compute composite pose loss.

        Args:
            predictions: Dict containing:
                - 'cls_logits': predicted class logits
                - 'pred_boxes': predicted bounding boxes (x1, y1, x2, y2)
                - 'pred_dfl': predicted DFL distribution tensor (optional)
                - 'pred_kpts': predicted keypoints (N, K, 2) or (N, K, 3)
            targets: Dict containing:
                - 'cls_targets': target labels
                - 'target_boxes': target bounding boxes (x1, y1, x2, y2)
                - 'target_dfl': target DFL distribution (optional)
                - 'target_kpts': target keypoints (N, K, 2)
                - 'target_vis': target keypoint visibility (N, K)

        Returns:
            Tuple of (total_loss_tensor, breakdown_dict).
        """
        if torch is None:
            return 0.0, {"loss_total": 0.0}

        # 1. Detection Base Loss (Box + Cls + DFL)
        det_loss, breakdown = self.detection_loss(predictions, targets)

        # 2. Keypoint Localization Loss (OKS + L1)
        pred_kpts = predictions.get("pred_kpts")
        target_kpts = targets.get("target_kpts")
        target_vis = targets.get("target_vis")
        target_boxes = targets.get("target_boxes")

        kpt_loss_val = torch.tensor(0.0, device=det_loss.device if hasattr(det_loss, "device") else "cpu")
        vis_loss_val = torch.tensor(0.0, device=det_loss.device if hasattr(det_loss, "device") else "cpu")

        if pred_kpts is not None and target_kpts is not None:
            # If pred_kpts has 3 channels (x, y, vis_logit)
            if pred_kpts.shape[-1] == 3:
                coords_pred = pred_kpts[..., :2]
                vis_logits = pred_kpts[..., 2]
            else:
                coords_pred = pred_kpts
                vis_logits = None

            # Calculate box area for OKS
            if target_boxes is not None and len(target_boxes) > 0:
                bw = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1e-3)
                bh = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1e-3)
                areas = bw * bh
            else:
                areas = torch.ones((len(target_kpts), 1), device=target_kpts.device)

            oks = self.oks_loss(coords_pred, target_kpts, areas, target_vis)
            l1 = self.kpt_l1_loss(coords_pred, target_kpts, target_vis)
            kpt_loss_val = oks + 0.5 * l1

            # Visibility BCE loss
            if vis_logits is not None and target_vis is not None and F is not None:
                vis_binary = (target_vis > 0).float()
                vis_loss_val = F.binary_cross_entropy_with_logits(vis_logits, vis_binary)

        total_loss = det_loss + self.kpt_weight * kpt_loss_val + self.kpt_vis_weight * vis_loss_val

        breakdown["loss_kpt"] = float(kpt_loss_val.detach().item()) if hasattr(kpt_loss_val, "item") else 0.0
        breakdown["loss_kpt_vis"] = float(vis_loss_val.detach().item()) if hasattr(vis_loss_val, "item") else 0.0
        breakdown["loss_total"] = float(total_loss.detach().item()) if hasattr(total_loss, "item") else 0.0

        return total_loss, breakdown


__all__ = ["COCO_SIGMAS", "OKSLoss", "KeypointL1Loss", "AethelPoseLoss"]
