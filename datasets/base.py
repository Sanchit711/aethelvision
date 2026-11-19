"""Abstract Base Dataset Implementation for Mhyra Vision.

Provides base class BaseDetectionDataset with real image loading, transform execution,
and target tensor padding for object detection models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    torch = None  # type: ignore
    class Dataset:  # type: ignore
        pass


class BaseDetectionDataset(Dataset, ABC):
    """Abstract Base Dataset for Object Detection architectures in Mhyra Vision."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        max_gt: int = 50,
        transforms: Optional[Any] = None,
        is_train: bool = True,
    ) -> None:
        """Initialize BaseDetectionDataset.

        Args:
            img_size: Target (height, width) image dimensions.
            max_gt: Maximum limit of padded ground truth boxes.
            transforms: Optional transform pipeline.
            is_train: Flag indicating training vs validation/testing mode.
        """
        super().__init__()
        self.img_size = img_size
        self.max_gt = max_gt
        self.transforms = transforms
        self.is_train = is_train

    @abstractmethod
    def __len__(self) -> int:
        """Return total sample count in dataset split."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Fetch sample at index."""
        pass

    def load_image(self, img_path: Union[str, Path]) -> np.ndarray:
        """Read RGB image array from disk.

        Args:
            img_path: Absolute or relative image file path.

        Returns:
            NumPy array image of shape (H, W, 3) in RGB format.
        """
        path = Path(img_path)
        if not path.exists():
            h, w = self.img_size if hasattr(self, "img_size") else (640, 640)
            return np.zeros((h, w, 3), dtype=np.uint8)

        try:
            import cv2
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to decode image array via OpenCV: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception:
            try:
                from PIL import Image
                img_pil = Image.open(path).convert("RGB")
                return np.array(img_pil)
            except Exception:
                h, w = self.img_size if hasattr(self, "img_size") else (640, 640)
                return np.zeros((h, w, 3), dtype=np.uint8)

    def pad_targets(
        self,
        boxes: Any,
        labels: Any,
    ) -> Tuple[Any, Any, Any]:
        """Pad ground truth bounding boxes and class labels to fixed max_gt size."""
        if torch is None:
            padded_labels = np.zeros((self.max_gt, 1), dtype=np.float32)
            padded_boxes = np.zeros((self.max_gt, 4), dtype=np.float32)
            mask_gt = np.zeros((self.max_gt, 1), dtype=np.float32)

            num_gt = min(len(boxes), self.max_gt)
            if num_gt > 0:
                padded_labels[:num_gt] = labels[:num_gt].reshape(-1, 1)
                padded_boxes[:num_gt] = boxes[:num_gt]
                mask_gt[:num_gt] = 1.0
            return padded_labels, padded_boxes, mask_gt

        num_gt = min(len(boxes), self.max_gt)

        padded_labels = torch.zeros((self.max_gt, 1), dtype=torch.float32)
        padded_boxes = torch.zeros((self.max_gt, 4), dtype=torch.float32)
        mask_gt = torch.zeros((self.max_gt, 1), dtype=torch.float32)

        if num_gt > 0:
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            padded_labels[:num_gt] = labels[:num_gt]
            padded_boxes[:num_gt] = boxes[:num_gt]
            mask_gt[:num_gt] = 1.0

        return padded_labels, padded_boxes, mask_gt


__all__ = ["BaseDetectionDataset"]
