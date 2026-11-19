"""Factory Interface for Object Detection Datasets and DataLoaders in Mhyra Vision."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from datasets.base import BaseDetectionDataset
from datasets.coco import COCODataset
from datasets.voc import VOCDataset
from datasets.yolo import YOLODataset


class DatasetFactory:
    """Universal Dataset Factory for instantiating dataset split objects."""

    @staticmethod
    def create_dataset(
        data_path_or_manifest: Union[str, Path],
        dataset_type: str = "auto",
        split: str = "train",
        img_size: Tuple[int, int] = (640, 640),
        max_gt: int = 50,
        transforms: Optional[Any] = None,
        is_train: bool = True,
        **kwargs: Any,
    ) -> BaseDetectionDataset:
        """Create and return instantiated BaseDetectionDataset instance.

        Args:
            data_path_or_manifest: Path to dataset folder or manifest file.
            dataset_type: Format string ('auto', 'yolo', 'coco', 'voc').
            split: Dataset split ('train', 'val', 'test').
            img_size: Target (height, width) tuple.
            max_gt: Maximum limit of padded ground truth boxes.
            transforms: Optional transform pipeline.
            is_train: Flag indicating training vs validation mode.

        Returns:
            Instantiated BaseDetectionDataset concrete object.
        """
        path = Path(data_path_or_manifest)
        type_key = dataset_type.lower()

        # Auto-detect format type if requested
        if type_key == "auto":
            if path.is_file() and path.suffix in (".json",):
                type_key = "coco"
            elif path.is_dir() and (path / "JPEGImages").exists():
                type_key = "voc"
            else:
                type_key = "yolo"

        if type_key == "yolo":
            return YOLODataset(
                data_path_or_yaml=path,
                split=split,
                img_size=img_size,
                max_gt=max_gt,
                transforms=transforms,
                is_train=is_train,
            )
        elif type_key == "coco":
            ann_file = kwargs.get("annotation_json", path)
            img_dir = kwargs.get("images_dir", path.parent if path.is_file() else path)
            return COCODataset(
                annotation_json=ann_file,
                images_dir=img_dir,
                img_size=img_size,
                max_gt=max_gt,
                transforms=transforms,
                is_train=is_train,
            )
        elif type_key == "voc":
            return VOCDataset(
                data_dir=path,
                split=split,
                img_size=img_size,
                max_gt=max_gt,
                transforms=transforms,
                is_train=is_train,
            )
        else:
            # Fallback to YOLODataset
            return YOLODataset(
                data_path_or_yaml=path,
                split=split,
                img_size=img_size,
                max_gt=max_gt,
                transforms=transforms,
                is_train=is_train,
            )


__all__ = ["DatasetFactory"]
