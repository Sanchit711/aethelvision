"""Distribution Focal Loss (DFL) for Aethel Detector.

Computes Distribution Focal Loss over discrete bounding box distance regression probability bins.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) module."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize DistributionFocalLoss.

        Args:
            reg_max: Max regression bin count integer (default 16).
        """
        super().__init__()
        self.reg_max = reg_max

    def forward(
        self,
        pred_dist: torch.Tensor,
        target_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DFL loss between predicted probability distributions and continuous target distances.

        Args:
            pred_dist: Predicted distribution logits of shape (N, 4, reg_max).
            target_dist: Target continuous distances tensor of shape (N, 4).

        Returns:
            Scalar DFL loss tensor.
        """
        if pred_dist.numel() == 0:
            return pred_dist.sum() * 0.0

        # Clamp target distances to [0, reg_max - 1.01]
        target_dist = target_dist.clamp(0, self.reg_max - 1.01)

        # Left and right discrete bin indices
        tl = target_dist.long()
        tr = tl + 1

        wl = tr.float() - target_dist
        wr = target_dist - tl.float()

        # Reshape for cross entropy
        pred_logits = pred_dist.view(-1, self.reg_max)
        tl_flat = tl.view(-1)
        tr_flat = tr.view(-1)

        loss_l = F.cross_entropy(pred_logits, tl_flat, reduction="none").view_as(target_dist)
        loss_r = F.cross_entropy(pred_logits, tr_flat, reduction="none").view_as(target_dist)

        dfl_loss = loss_l * wl + loss_r * wr
        return dfl_loss.mean()


__all__ = ["DistributionFocalLoss"]
