"""Classification and Focal Loss Functions for Aethel Vision Framework.

Provides Focal Loss and Binary Cross Entropy (BCE) with optional label smoothing.
"""

from typing import Any, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore
    F = None  # type: ignore

from aethel.losses.base import BaseLoss
from aethel.registry.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("FocalLoss")
class FocalLoss(BaseLoss):
    """Sigmoid Focal Loss for handling extreme foreground-background class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred_logits: Any, targets: Any) -> Tuple[Any, dict]:
        if torch is None or F is None or not hasattr(pred_logits, "shape"):
            return 0.0, {"focal_loss": 0.0}

        if pred_logits.shape[0] == 0:
            return pred_logits.sum() * 0.0, {"focal_loss": 0.0}

        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        mean_loss = loss.mean()
        return mean_loss, {"focal_loss": mean_loss.item()}


@LOSS_REGISTRY.register("BCEWithLogitsLoss")
class BCEWithLogitsLoss(BaseLoss):
    """Binary Cross Entropy with Logits Loss."""

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred_logits: Any, targets: Any) -> Tuple[Any, dict]:
        if torch is None or F is None or not hasattr(pred_logits, "shape"):
            return 0.0, {"bce_loss": 0.0}

        if pred_logits.shape[0] == 0:
            return pred_logits.sum() * 0.0, {"bce_loss": 0.0}

        loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="mean")
        return loss, {"bce_loss": loss.item()}


__all__ = ["FocalLoss", "BCEWithLogitsLoss"]
