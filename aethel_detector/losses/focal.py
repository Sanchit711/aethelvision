"""Classification Loss Functions for Aethel Detector.

Provides FocalLoss and BCEClassificationLoss modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEClassificationLoss(nn.Module):
    """Binary Cross-Entropy Classification Loss module."""

    def forward(self, pred_logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """Compute BCE classification loss with logits.

        Args:
            pred_logits: Predicted class logits tensor (N, num_classes).
            target_labels: Target class one-hot or probability labels (N, num_classes).

        Returns:
            Scalar loss tensor.
        """
        if pred_logits.numel() == 0:
            return pred_logits.sum() * 0.0

        return F.binary_cross_entropy_with_logits(pred_logits, target_labels)


class FocalLoss(nn.Module):
    """Focal Loss module for class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """Initialize FocalLoss.

        Args:
            alpha: Balancing weighting factor float (default 0.25).
            gamma: Focusing exponent factor float (default 2.0).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss.

        Args:
            pred_logits: Predicted class logits tensor (N, num_classes).
            target_labels: Target class one-hot or probability labels (N, num_classes).

        Returns:
            Scalar loss tensor.
        """
        if pred_logits.numel() == 0:
            return pred_logits.sum() * 0.0

        bce = F.binary_cross_entropy_with_logits(pred_logits, target_labels, reduction="none")
        p = torch.sigmoid(pred_logits)
        p_t = p * target_labels + (1.0 - p) * (1.0 - target_labels)
        alpha_factor = self.alpha * target_labels + (1.0 - self.alpha) * (1.0 - target_labels)
        modulating_factor = (1.0 - p_t) ** self.gamma

        focal_loss = alpha_factor * modulating_factor * bce
        return focal_loss.mean()


__all__ = ["BCEClassificationLoss", "FocalLoss"]
