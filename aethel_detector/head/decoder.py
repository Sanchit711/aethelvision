"""Detection Head Output Decoder for Aethel Detector.

Transforms multi-scale raw head predictions into decoded spatial bounding box
coordinates and class probability scores.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class HeadDecoder(nn.Module):
    """Decodes multi-scale raw prediction tensors into spatial bounding boxes."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize HeadDecoder.

        Args:
            reg_max: DFL regression max bin bound integer (default 16).
        """
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "proj",
            torch.arange(reg_max, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        raw_outputs: List[torch.Tensor],
        strides: Tuple[int, int, int] = (8, 16, 32),
    ) -> torch.Tensor:
        """Decode multi-scale raw predictions into a flattened tensor of bounding boxes and scores.

        Args:
            raw_outputs: List of raw prediction tensors from DecoupledDetectHead.
            strides: Scale strides tuple (8, 16, 32).

        Returns:
            Decoded prediction tensor of shape (B, Total_Anchors, 4 + num_classes).
            Spatial box coordinates are [x1, y1, x2, y2].
        """
        decoded_list = []
        for i, pred in enumerate(raw_outputs):
            batch_size, channels, h, w = pred.shape
            stride = strides[i]

            # Split classification and box regression channels
            cls_logits = pred[:, :-4 * self.reg_max, :, :]
            reg_logits = pred[:, -4 * self.reg_max:, :, :]

            # Compute class probabilities
            cls_scores = torch.sigmoid(cls_logits)

            # Reshape predictions for spatial decoding
            cls_scores = cls_scores.flatten(2).permute(0, 2, 1)  # (B, H*W, num_classes)
            reg_logits = (
                reg_logits.view(batch_size, 4, self.reg_max, h * w)
                .permute(0, 3, 1, 2)
            )  # (B, H*W, 4, reg_max)

            # Softmax expectation over DFL bins
            reg_prob = torch.softmax(reg_logits, dim=-1)
            dist = torch.matmul(reg_prob, self.proj.to(pred.device)) * stride  # (B, H*W, 4)

            # Generate grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=pred.device, dtype=torch.int64).to(torch.float32),
                torch.arange(w, device=pred.device, dtype=torch.int64).to(torch.float32),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).view(1, h * w, 2).to(dtype=pred.dtype) * stride  # (1, H*W, 2)

            # Distance decoding: x1 = x_center - d0, y1 = y_center - d1, etc.
            x1y1 = grid - dist[:, :, :2]
            x2y2 = grid + dist[:, :, 2:]
            boxes = torch.cat([x1y1, x2y2], dim=-1)  # (B, H*W, 4)

            # Concatenate decoded boxes [x1, y1, x2, y2] and class scores
            decoded = torch.cat([boxes, cls_scores], dim=-1)  # (B, H*W, 4 + num_classes)
            decoded_list.append(decoded)

        return torch.cat(decoded_list, dim=1)  # (B, Total_Anchors, 4 + num_classes)


__all__ = ["HeadDecoder"]
