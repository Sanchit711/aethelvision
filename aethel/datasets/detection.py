"""Production Object Detection Dataset Implementation for Aethel Vision.

Universal, model-agnostic dataset implementation supporting YOLO TXT, Pascal VOC XML,
and COCO JSON formats with lazy loading, caching, and transform integration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from aethel.datasets.base import BaseDataset
from aethel.datasets.cache import DatasetCache, DiskDatasetCache
from aethel.datasets.loader import ImageLoader
from aethel.datasets.metadata import AnnotationInfo, DatasetMetadata, ImageInfo
from aethel.datasets.parser import DatasetParser, LabelParser
from aethel.datasets.statistics import DatasetStatistics
from aethel.datasets.transforms import TransformPipeline
from aethel.registry.registry import DATASET_REGISTRY
from aethel.utils.exceptions import DatasetException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.detection")


@DATASET_REGISTRY.register("DetectionDataset")
class DetectionDataset(BaseDataset):
    """Universal production detection dataset implementation for Aethel Vision."""

    def __init__(
        self,
        root: Union[str, Path],
        format_type: str = "auto",
        split: str = "train",
        config: Optional[Dict[str, Any]] = None,
        transforms: Optional[TransformPipeline] = None,
        is_train: bool = True,
        use_cache: bool = False,
    ) -> None:
        """Initialize DetectionDataset.

        Args:
            root: Root path to dataset directory.
            format_type: Dataset format string ('auto', 'yolo', 'voc', 'coco').
            split: Dataset split ('train', 'val', 'test').
            config: Configuration dictionary.
            transforms: Optional TransformPipeline instance.
            is_train: Training vs validation flag.
            use_cache: Master flag to enable in-memory image buffer caching.
        """
        super().__init__(root=root, transforms=transforms, is_train=is_train)
        self.split = split
        self.config = config or {}

        # Resolve directory paths (supports configurable folder names via YAML)
        images_dir_name = self.config.get("images_dir", "images")
        labels_dir_name = self.config.get("labels_dir", "labels")

        split_dir = self.root / split if (self.root / split).exists() else self.root
        self.images_dir = split_dir / images_dir_name if (split_dir / images_dir_name).exists() else split_dir
        self.labels_dir = split_dir / labels_dir_name if (split_dir / labels_dir_name).exists() else split_dir

        # Format detection
        if format_type == "auto":
            self.format_type = DatasetParser.detect_format(self.root)
        else:
            self.format_type = format_type.lower()

        # Cache setup
        self.cache = DatasetCache(enabled=use_cache)

        # Parse samples metadata with disk cache (.cache) acceleration
        self.cache_file = self.root / f".{self.split}_dataset.cache"
        self.samples: List[ImageInfo] = self._discover_and_parse_samples()

        # Class names setup
        self.class_names: List[str] = self.config.get("class_names", [])

        # Transform pipeline fallback
        if self.transforms is None:
            self.transforms = TransformPipeline(config=self.config, is_train=is_train)

        logger.info(f"Loaded DetectionDataset '{self.format_type.upper()}' [{split}]: {len(self.samples)} samples found at {self.root}")

    def _discover_and_parse_samples(self) -> List[ImageInfo]:
        """Discover image files and link corresponding annotations with disk cache acceleration."""
        if not self.images_dir.exists() and not self.root.exists():
            raise DatasetException(
                message=f"Dataset images directory does not exist: {self.images_dir}",
                details=f"Root path: {self.root}",
            )

        # Phase 5: Check persistent disk cache
        signature = DiskDatasetCache.compute_signature(self.root, self.split)
        cached_samples = DiskDatasetCache.load_samples(self.cache_file, signature)
        if cached_samples is not None:
            return cached_samples

        if self.format_type == "coco":
            json_file = self.root / f"annotations_{self.split}.json"
            if not json_file.exists():
                json_file = self.root / "annotations.json"
            if json_file.exists():
                samples, _ = DatasetParser.parse_coco_json(json_file)
                DiskDatasetCache.save_samples(self.cache_file, signature, samples)
                return samples

        # YOLO / VOC file discovery
        samples: List[ImageInfo] = []
        if self.images_dir.exists():
            image_paths = sorted([p for p in self.images_dir.rglob("*") if ImageLoader.is_valid_image_extension(p)])

            for img_path in image_paths:
                # Locate label file
                rel_path = img_path.relative_to(self.images_dir)
                label_path = (self.labels_dir / rel_path).with_suffix(".txt")

                anns: List[AnnotationInfo] = []
                if self.format_type == "yolo" and label_path.exists():
                    anns = LabelParser.parse_yolo_file(label_path)
                elif self.format_type == "voc":
                    xml_path = (self.labels_dir / rel_path).with_suffix(".xml")
                    if xml_path.exists():
                        class_map = {name: i for i, name in enumerate(self.config.get("class_names", []))}
                        _, anns = LabelParser.parse_voc_file(xml_path, class_map)

                samples.append(
                    ImageInfo(
                        image_path=img_path,
                        label_path=label_path if label_path.exists() else None,
                        annotations=anns,
                    )
                )

        # Save to persistent disk cache
        if samples:
            DiskDatasetCache.save_samples(self.cache_file, signature, samples)

        return samples


    def __len__(self) -> int:
        """Return total sample count."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[AnnotationInfo]]:
        """Fetch sample at given index with lazy loading, caching, and transforms.

        Args:
            idx: Sample index integer.

        Returns:
            Tuple of (transformed_image_array, annotations_list).
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self.samples) - 1}]")

        sample = self.samples[idx]

        # 1. Fetch image from cache or disk with fallback on corrupt image
        img = self.cache.get(str(sample.image_path))
        if img is None:
            try:
                img = ImageLoader.load_image(sample.image_path, mode="RGB")
                self.cache.put(str(sample.image_path), img)
            except Exception as e:
                logger.warning(f"Corrupt or unreadable image at {sample.image_path}: {e}. Falling back to next sample...")
                next_idx = (idx + 1) % len(self.samples)
                return self.__getitem__(next_idx)

        # Update height and width in metadata
        if sample.height == 0 or sample.width == 0:
            sample.height, sample.width = img.shape[0], img.shape[1]

        # 2. Clone annotations
        anns = [AnnotationInfo(class_id=a.class_id, bbox=a.bbox, format=a.format) for a in sample.annotations]

        # 3. Apply transform pipeline
        if self.transforms is not None:
            img, anns = self.transforms(img, anns)

        return img, anns

    def get_stats(self) -> Dict[str, Any]:
        """Return quantitative dataset statistics."""
        return DatasetStatistics.compute(self.samples, class_names=self.class_names)


__all__ = ["DetectionDataset"]
