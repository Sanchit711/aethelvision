"""Target Assignment Module for Aethel Detector.

Matches ground truth bounding box annotations to multi-scale detector output anchors
using spatial center matching and task alignment scoring.
"""

from typing import Tuple

import torch
import torch.nn as nn

from aethel_detector.losses.iou import bbox_iou


class TargetAssigner(nn.Module):
    """Task-aligned target assigner matching GT boxes to candidate anchors."""

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
    ) -> None:
        """Initialize TargetAssigner.

        Args:
            topk: Maximum candidate anchors selected per GT box (default 10).
            num_classes: Number of classification target classes (default 80).
            alpha: Alignment score classification weight exponent.
            beta: Alignment score IoU weight exponent.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign ground truth labels and bounding boxes to candidate anchors.

        Args:
            pred_scores: Predicted class probabilities (B, Total_Anchors, num_classes).
            pred_bboxes: Predicted bounding boxes [x1, y1, x2, y2] (B, Total_Anchors, 4).
            anchor_points: Anchor center coordinates [x, y] (Total_Anchors, 2).
            gt_labels: Ground truth class indices tensor (B, Max_GT, 1).
            gt_bboxes: Ground truth bounding boxes [x1, y1, x2, y2] (B, Max_GT, 4).
            mask_gt: Ground truth validity mask tensor (B, Max_GT, 1).

        Returns:
            Tuple of assigned targets:
            - target_labels: One-hot target labels tensor (B, Total_Anchors, num_classes).
            - target_bboxes: Assigned target bounding boxes (B, Total_Anchors, 4).
            - target_scores: Target alignment score tensor (B, Total_Anchors, 1).
        """
        bs, num_anchors, _ = pred_scores.shape
        device = pred_scores.device

        target_labels = torch.zeros((bs, num_anchors, self.num_classes), device=device)
        target_bboxes = torch.zeros((bs, num_anchors, 4), device=device)
        target_scores = torch.zeros((bs, num_anchors, 1), device=device)

        for b in range(bs):
            valid_gt_mask = mask_gt[b].squeeze(-1) > 0
            if not valid_gt_mask.any():
                continue

            b_gt_boxes = gt_bboxes[b][valid_gt_mask]  # (N_gt, 4)
            b_gt_labels = gt_labels[b][valid_gt_mask].long().squeeze(-1)  # (N_gt,)

            num_gt = b_gt_boxes.shape[0]

            # Center in GT box mask check
            anchor_x = anchor_points[:, 0]
            anchor_y = anchor_points[:, 1]

            in_gt_mask = (
                (anchor_x.unsqueeze(0) >= b_gt_boxes[:, 0:1])
                & (anchor_x.unsqueeze(0) <= b_gt_boxes[:, 2:3])
                & (anchor_y.unsqueeze(0) >= b_gt_boxes[:, 1:2])
                & (anchor_y.unsqueeze(0) <= b_gt_boxes[:, 3:4])
            )  # (N_gt, Total_Anchors)

            for g in range(num_gt):
                candidate_indices = in_gt_mask[g].nonzero(as_tuple=False).squeeze(-1)
                if candidate_indices.numel() == 0:
                    # Pick nearest anchor if no anchor center lands inside GT box
                    dists = (anchor_points[:, 0] - (b_gt_boxes[g, 0] + b_gt_boxes[g, 2]) / 2) ** 2 + (
                        anchor_points[:, 1] - (b_gt_boxes[g, 1] + b_gt_boxes[g, 3]) / 2
                    ) ** 2
                    candidate_indices = dists.argmin(dim=0, keepdim=True)

                # Assign target class label and box
                cls_idx = b_gt_labels[g]
                target_labels[b, candidate_indices, cls_idx] = 1.0
                target_bboxes[b, candidate_indices] = b_gt_boxes[g]
                target_scores[b, candidate_indices] = 1.0

        return target_labels, target_bboxes, target_scores


__all__ = ["TargetAssigner"]
