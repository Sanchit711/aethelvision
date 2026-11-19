"""Dataset Metadata and Data Structures for Aethel Vision.

Defines immutable container classes for image metadata, bounding box annotations,
and dataset summary specifications.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class AnnotationInfo:
    """Bounding box annotation data structure.

    Attributes:
        class_id: Target class index integer.
        bbox: Bounding box coordinates tuple (x1, y1, x2, y2) in pixel dimensions or normalized float.
        format: Coordinate format ('xyxy', 'xywh_norm', 'xywh_abs').
        area: Optional bounding box area float.
        is_crowd: Flag indicating COCO crowd region.
        score: Confidence score float (for pseudo-labels or predictions).
    """

    class_id: int
    bbox: Tuple[float, float, float, float]
    format: str = "xyxy"
    area: Optional[float] = None
    is_crowd: bool = False
    score: float = 1.0

    def to_xyxy(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert bounding box to absolute [x1, y1, x2, y2] pixel coordinates."""
        if self.format == "xyxy":
            return self.bbox
        elif self.format == "xywh_norm":
            xc, yc, w, h = self.bbox
            x1 = (xc - w / 2.0) * img_width
            y1 = (yc - h / 2.0) * img_height
            x2 = (xc + w / 2.0) * img_width
            y2 = (yc + h / 2.0) * img_height
            return (x1, y1, x2, y2)
        elif self.format == "xywh_abs":
            x, y, w, h = self.bbox
            return (x, y, x + w, y + h)
        return self.bbox


@dataclass
class ImageInfo:
    """Metadata container for a single image sample.

    Attributes:
        image_path: Absolute filepath to image asset.
        height: Image pixel height.
        width: Image pixel width.
        channels: Color channel count (default 3 for RGB).
        label_path: Optional path to text/XML label file.
        annotations: List of AnnotationInfo objects.
    """

    image_path: Path
    height: int = 0
    width: int = 0
    channels: int = 3
    label_path: Optional[Path] = None
    annotations: List[AnnotationInfo] = field(default_factory=list)


@dataclass
class DatasetMetadata:
    """High-level metadata summary for a vision dataset.

    Attributes:
        name: Dataset identifier name.
        format_type: Format string ('yolo', 'voc', 'coco', 'aethel').
        num_classes: Total class count integer.
        class_names: List of class label strings.
        total_images: Total image sample count.
        total_annotations: Total bounding box instance count.
        splits: Supported split names list (['train', 'val', 'test']).
    """

    name: str = "aethel_dataset"
    format_type: str = "auto"
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    total_images: int = 0
    total_annotations: int = 0
    splits: List[str] = field(default_factory=lambda: ["train", "val"])


__all__ = ["AnnotationInfo", "ImageInfo", "DatasetMetadata"]
