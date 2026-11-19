"""Abstract Base Dataset Class for Aethel Vision Framework."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    from torch.utils.data import Dataset
except Exception:
    class Dataset:  # type: ignore
        """Fallback Dataset base class when PyTorch native C-DLLs are constrained."""
        pass


class BaseDataset(Dataset, ABC):
    """Abstract base dataset interface for object detection and computer vision tasks."""

    def __init__(
        self,
        root: Union[str, Path],
        transforms: Optional[Any] = None,
        is_train: bool = True,
    ) -> None:
        """Initialize BaseDataset.

        Args:
            root: Root path to the dataset directory.
            transforms: Optional data augmentation or preprocessing pipeline.
            is_train: Flag indicating training vs validation mode.
        """
        super().__init__()
        self.root = Path(root)
        self.transforms = transforms
        self.is_train = is_train

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Fetch sample at given index.

        Returns:
            Tuple of (image_tensor, target_annotations).
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics summary dictionary."""
        pass


__all__ = ["BaseDataset"]
