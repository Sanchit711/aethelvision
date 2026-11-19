"""Objectness Loss Module for Aethel Detector.

Computes binary cross-entropy objectness loss for anchor feature locations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectnessLoss(nn.Module):
    """Binary Cross-Entropy Objectness Loss module."""

    def __init__(self, pos_weight: float = 1.0) -> None:
        """Initialize ObjectnessLoss.

        Args:
            pos_weight: Positive anchor weight scaling float.
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred_obj: torch.Tensor, target_obj: torch.Tensor) -> torch.Tensor:
        """Compute objectness loss.

        Args:
            pred_obj: Predicted objectness logits tensor (N, 1).
            target_obj: Target objectness binary labels tensor (N, 1).

        Returns:
            Scalar loss tensor.
        """
        if pred_obj.numel() == 0:
            return pred_obj.sum() * 0.0

        pos_w = torch.tensor([self.pos_weight], device=pred_obj.device)
        return F.binary_cross_entropy_with_logits(pred_obj, target_obj, pos_weight=pos_w)


__all__ = ["ObjectnessLoss"]
