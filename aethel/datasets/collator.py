"""Batch Collation Subsystem for Aethel Vision DataLoader.

Collates individual image and target annotation samples into unified PyTorch batch tensors.
"""

from typing import Any, Dict, List, Tuple

try:
    import torch
except Exception:
    torch = None  # type: ignore


class BatchCollator:
    """Collate function for PyTorch DataLoader in object detection tasks."""

    def __init__(self, format: str = "batch_idx") -> None:
        """Initialize BatchCollator.

        Args:
            format: Target tensor format ('batch_idx' -> [batch_idx, class_id, x1, y1, x2, y2]).
        """
        self.format = format

    def __call__(self, batch: List[Tuple[Any, List[Any]]]) -> Tuple[Any, Any]:
        """Collate batch samples into tensors.

        Args:
            batch: List of tuples (image_array_or_tensor, annotations_list).

        Returns:
            Tuple of (images_tensor, targets_tensor).
        """
        images = [item[0] for item in batch]
        annotations_list = [item[1] for item in batch]

        if torch is not None:
            # Convert images to torch FloatTensor (B, C, H, W) normalized to [0, 1]
            torch_imgs = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    torch_imgs.append(img)
                else:
                    # NumPy (H, W, C) -> PyTorch (C, H, W)
                    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    torch_imgs.append(t)

            batch_images = torch.stack(torch_imgs, dim=0)

            # Collate targets into (Total_Objects, 6) -> [batch_idx, class_id, x1, y1, x2, y2]
            target_list = []
            for batch_idx, anns in enumerate(annotations_list):
                for ann in anns:
                    if hasattr(ann, "class_id"):
                        cid = ann.class_id
                        bbox = ann.bbox
                    elif isinstance(ann, dict):
                        cid = ann.get("class_id", ann.get("category_id", 0))
                        bbox = ann.get("bbox", ann.get("boxes", [0, 0, 0, 0]))
                    elif isinstance(ann, (tuple, list)):
                        cid = ann[0]
                        bbox = ann[1:]
                    else:
                        cid = 0
                        bbox = [0, 0, 0, 0]
                    target_list.append([batch_idx, float(cid), float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

            if target_list:
                batch_targets = torch.tensor(target_list, dtype=torch.float32)
            else:
                batch_targets = torch.zeros((0, 6), dtype=torch.float32)

            return batch_images, batch_targets
        else:
            return images, annotations_list



detection_collate_fn = BatchCollator()

__all__ = ["BatchCollator", "detection_collate_fn"]
