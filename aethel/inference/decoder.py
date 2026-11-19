"""Multi-Scale Prediction Decoder and Anchor Grid Generator for Aethel Vision Framework.

Decodes raw multi-scale head outputs (DFL distribution logits and classification logits)
into candidate bounding boxes [x1, y1, x2, y2] and class probabilities.
"""

from typing import Any, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore
    F = None  # type: ignore

from aethel.utils.logger import get_logger

logger = get_logger("aethel.inference.decoder")


class GridGenerator:
    """Generates multi-scale spatial grid anchors for dynamic image dimensions."""

    @staticmethod
    def generate_anchors(
        feat_shapes: List[Tuple[int, int]],
        strides: Tuple[int, ...] = (8, 16, 32),
    ) -> Tuple[Any, Any]:
        """Generate anchor grid coordinates and stride vectors for pyramid feature levels.

        Args:
            feat_shapes: List of (H_i, W_i) spatial dimensions per scale.
            strides: Feature scale spatial strides (default 8, 16, 32).

        Returns:
            Tuple of (grid_anchors, stride_tensor).
            - grid_anchors: Shape (Total_Anchors, 2) grid center offsets [grid_x, grid_y].
            - stride_tensor: Shape (Total_Anchors, 1) scale stride values.
        """
        if torch is None:
            return None, None

        anchor_grids = []
        stride_vectors = []

        for (h, w), stride in zip(feat_shapes, strides):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            stride_vec = torch.full((h * w, 1), float(stride), dtype=torch.float32)

            anchor_grids.append(grid)
            stride_vectors.append(stride_vec)

        anchor_tensor = torch.cat(anchor_grids, dim=0)
        strides_tensor = torch.cat(stride_vectors, dim=0)
        return anchor_tensor, strides_tensor


class DistributionFocalDecoder:
    """Decodes DFL discrete probability distribution bins into continuous distance offsets."""

    def __init__(self, reg_max: int = 16) -> None:
        self.reg_max = reg_max
        if torch is not None:
            self.project = torch.linspace(0, reg_max - 1, reg_max, dtype=torch.float32)
        else:
            self.project = None

    def decode(self, dfl_logits: Any) -> Any:
        """Decode DFL logits (N, 4 * reg_max) into 4 continuous distance offsets (N, 4)."""
        if torch is None or F is None:
            return dfl_logits

        n = dfl_logits.shape[0]
        dfl_reshaped = dfl_logits.view(n, 4, self.reg_max)
        probs = F.softmax(dfl_reshaped, dim=-1)

        proj = self.project.to(dfl_logits.device)
        distances = (probs * proj).sum(dim=-1)
        return distances


class PredictionDecoder:
    """Primary multi-scale prediction decoder."""

    def __init__(self, num_classes: int = 80, reg_max: int = 16, strides: Tuple[int, ...] = (8, 16, 32)) -> None:
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.dfl_decoder = DistributionFocalDecoder(reg_max=reg_max)

    def decode(self, raw_outputs: List[Any]) -> Tuple[Any, Any]:
        """Decode multi-scale raw head prediction tensors.

        Args:
            raw_outputs: List of scale level tensors [N3, N4, N5].
                Each tensor has shape (B, num_classes + 4 * reg_max, H_i, W_i).

        Returns:
            Tuple of (decoded_boxes, decoded_scores).
            - decoded_boxes: Tensor of shape (B, Total_Anchors, 4) in [x1, y1, x2, y2] format.
            - decoded_scores: Tensor of shape (B, Total_Anchors, num_classes) sigmoid probabilities.
        """
        if torch is None:
            return None, None

        batch_size = raw_outputs[0].shape[0]
        feat_shapes = [(out.shape[2], out.shape[3]) for out in raw_outputs]

        # 1. Flatten and concatenate multi-scale scale prediction outputs
        flattened_list = []
        for out in raw_outputs:
            b, c, h, w = out.shape
            flat = out.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
            flattened_list.append(flat)

        concat_preds = torch.cat(flattened_list, dim=1)  # (B, Total_Anchors, C)

        # Split into classification logits and DFL regression logits
        cls_logits = concat_preds[:, :, : self.num_classes]
        reg_logits = concat_preds[:, :, self.num_classes :]

        # 2. Compute classification sigmoid probabilities
        scores = torch.sigmoid(cls_logits)

        # 3. Generate anchor grid coordinates and strides
        grid_anchors, strides_tensor = GridGenerator.generate_anchors(feat_shapes, self.strides)
        grid_anchors = grid_anchors.to(concat_preds.device)
        strides_tensor = strides_tensor.to(concat_preds.device)

        # 4. Decode DFL distance offsets for each batch item
        total_anchors = concat_preds.shape[1]
        distances = self.dfl_decoder.decode(reg_logits.view(-1, 4 * self.reg_max)).view(batch_size, total_anchors, 4)

        # Multiply distances by scale stride
        distances = distances * strides_tensor.unsqueeze(0)

        # 5. Convert grid centers + distances -> absolute bounding boxes [x1, y1, x2, y2]
        anchor_centers = (grid_anchors + 0.5) * strides_tensor  # (Total_Anchors, 2)
        cx = anchor_centers[:, 0].unsqueeze(0)
        cy = anchor_centers[:, 1].unsqueeze(0)

        dx1 = distances[:, :, 0]
        dy1 = distances[:, :, 1]
        dx2 = distances[:, :, 2]
        dy2 = distances[:, :, 3]

        x1 = cx - dx1
        y1 = cy - dy1
        x2 = cx + dx2
        y2 = cy + dy2

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return boxes, scores


AnchorGenerator = GridGenerator
ClassificationDecoder = PredictionDecoder
ObjectnessDecoder = PredictionDecoder
DistributionFocalDecoder = DistributionFocalDecoder

__all__ = [
    "GridGenerator",
    "AnchorGenerator",
    "DistributionFocalDecoder",
    "PredictionDecoder",
    "ClassificationDecoder",
    "ObjectnessDecoder",
]
