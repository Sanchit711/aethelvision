"""Distribution Focal Loss (DFL) for Aethel Vision Framework.

Optimizes continuous distance bounding box regression over discrete probability bins.
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


@LOSS_REGISTRY.register("DistributionFocalLoss")
class DistributionFocalLoss(BaseLoss):
    """Distribution Focal Loss (DFL) module."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: Any, target_dist: Any) -> Tuple[Any, dict]:
        if torch is None or F is None or not hasattr(pred_dist, "shape"):
            return 0.0, {"dfl_loss": 0.0}

        if pred_dist.shape[0] == 0:
            return pred_dist.sum() * 0.0, {"dfl_loss": 0.0}

        dis_left = target_dist.long().clamp(0, self.reg_max - 2)
        dis_right = dis_left + 1
        wl = dis_right.float() - target_dist
        wr = target_dist - dis_left.float()

        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max), dis_left.view(-1), reduction="none")
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max), dis_right.view(-1), reduction="none")

        loss = (loss_left.view_as(target_dist) * wl + loss_right.view_as(target_dist) * wr).mean()
        return loss, {"dfl_loss": loss.item()}


__all__ = ["DistributionFocalLoss"]
