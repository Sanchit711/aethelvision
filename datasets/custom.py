"""Custom Format Dataset Loader for Mhyra Detector."""

from typing import Any, Dict, List, Tuple

import torch

from datasets.base import BaseDetectionDataset


class CustomDataset(BaseDetectionDataset):
    """Flexible custom dataset loader accepting a list of sample dictionaries."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        img_size: Tuple[int, int] = (640, 640),
        max_gt: int = 50,
    ) -> None:
        """Initialize CustomDataset.

        Args:
            samples: List of dictionaries containing 'image', 'boxes', and 'labels'.
            img_size: Target (height, width) tuple.
            max_gt: Maximum padded ground truth objects limit.
        """
        super().__init__(img_size=img_size, max_gt=max_gt)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch item at index.

        Returns:
            Tuple of (image_tensor, padded_labels, padded_boxes, mask_gt).
        """
        h_out, w_out = self.img_size
        sample = self.samples[index]

        img_raw = sample.get("image")
        if isinstance(img_raw, torch.Tensor):
            image = img_raw
        else:
            image = torch.zeros((3, h_out, w_out), dtype=torch.float32)

        raw_boxes = sample.get("boxes")
        if isinstance(raw_boxes, torch.Tensor):
            boxes = raw_boxes
        elif raw_boxes:
            boxes = torch.tensor(raw_boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        raw_labels = sample.get("labels")
        if isinstance(raw_labels, torch.Tensor):
            labels = raw_labels
        elif raw_labels:
            labels = torch.tensor(raw_labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 1), dtype=torch.float32)

        padded_labels, padded_boxes, mask_gt = self.pad_targets(boxes, labels)
        return image, padded_labels, padded_boxes, mask_gt


__all__ = ["CustomDataset"]
