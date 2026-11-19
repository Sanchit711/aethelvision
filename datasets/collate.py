"""Production Batch Collation Function for Mhyra Vision Object Detection Datasets."""

from typing import Any, List, Tuple

try:
    import torch
except Exception:
    torch = None  # type: ignore


class DetectionCollate:
    """Collation callable for stacking image tensors and ground truth targets into batch tuples."""

    def __call__(
        self, batch: List[Tuple[Any, Any, Any, Any]]
    ) -> Tuple[Any, Any, Any, Any]:
        """Collate list of sample tuples into batch tensors.

        Args:
            batch: List of tuples (image_tensor, padded_labels, padded_boxes, mask_gt).

        Returns:
            Tuple of (batch_images, batch_labels, batch_boxes, batch_masks).
        """
        if torch is None:
            images = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            boxes = [item[2] for item in batch]
            masks = [item[3] for item in batch]
            return images, labels, boxes, masks

        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.stack([item[1] for item in batch], dim=0)
        boxes = torch.stack([item[2] for item in batch], dim=0)
        masks = torch.stack([item[3] for item in batch], dim=0)

        return images, labels, boxes, masks


def collate_fn(
    batch: List[Tuple[Any, Any, Any, Any]]
) -> Tuple[Any, Any, Any, Any]:
    """Functional collate_fn interface."""
    collator = DetectionCollate()
    return collator(batch)


__all__ = ["DetectionCollate", "collate_fn"]
