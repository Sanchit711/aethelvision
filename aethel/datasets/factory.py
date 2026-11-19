"""Dataset Factory Subsystem for Aethel Vision.

Provides factory design pattern for instantiating dataset objects from configuration or explicit arguments.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from aethel.datasets.base import BaseDataset
from aethel.datasets.detection import DetectionDataset
from aethel.registry.registry import DATASET_REGISTRY
from aethel.utils.exceptions import DatasetException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.factory")


class DatasetFactory:
    """Factory for creating computer vision dataset instances."""

    @staticmethod
    def create(
        root: Union[str, Path],
        dataset_type: str = "DetectionDataset",
        split: str = "train",
        is_train: bool = True,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseDataset:
        """Create a dataset instance.

        Args:
            root: Path to dataset root directory.
            dataset_type: Type of dataset registered in DATASET_REGISTRY.
            split: Dataset split ('train', 'val', 'test').
            is_train: Flag indicating training mode.
            config: Optional configuration dictionary.
            **kwargs: Additional keyword arguments passed to dataset constructor.

        Returns:
            Instantiated BaseDataset object.
        """
        config = config or {}
        
        # Check registry first
        if dataset_type in DATASET_REGISTRY:
            dataset_cls = DATASET_REGISTRY.get(dataset_type)
            return dataset_cls(root=root, split=split, is_train=is_train, config=config, **kwargs)
            
        if dataset_type.lower() in ("detection", "detectiondataset", "yolo", "voc", "coco"):
            return DetectionDataset(root=root, split=split, is_train=is_train, config=config, **kwargs)

        raise DatasetException(f"Unknown dataset type: {dataset_type}. Registered types: {list(DATASET_REGISTRY.keys())}")


__all__ = ["DatasetFactory"]
