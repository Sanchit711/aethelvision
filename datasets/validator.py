"""Production Dataset Validation Engine for Mhyra Detector Datasets.

Scans object detection datasets for structural integrity, missing files, corrupted images,
invalid class IDs, zero-area bounding boxes, and out-of-bounds spatial coordinates.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image


@dataclass
class ValidationReport:
    """Dataset structural validation report manifest."""

    total_images: int = 0
    valid_images: int = 0
    missing_images: List[str] = field(default_factory=list)
    missing_labels: List[str] = field(default_factory=list)
    corrupted_images: List[str] = field(default_factory=list)
    empty_labels: List[str] = field(default_factory=list)
    invalid_class_ids: List[Dict[str, Any]] = field(default_factory=list)
    zero_area_boxes: List[Dict[str, Any]] = field(default_factory=list)
    out_of_bounds_boxes: List[Dict[str, Any]] = field(default_factory=list)
    invalid_boxes: List[Dict[str, Any]] = field(default_factory=list)
    is_valid: bool = True


class DatasetValidator:
    """Automated validator for YOLO, COCO, VOC, and custom detection datasets."""

    def __init__(self, num_classes: int = 80, img_size: Tuple[int, int] = (640, 640)) -> None:
        """Initialize DatasetValidator.

        Args:
            num_classes: Expected maximum class index count (default 80).
            img_size: Expected target (height, width) spatial dimensions.
        """
        self.num_classes = num_classes
        self.img_size = img_size

    def validate_yolo_pair(
        self,
        img_path: Path,
        label_path: Path,
        report: ValidationReport,
    ) -> None:
        """Validate a single YOLO image and label file pair."""
        # 1. Verify Image File Existence and Readability
        if not img_path.exists():
            report.missing_images.append(str(img_path))
            report.is_valid = False
            return

        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                w_img, h_img = img.size
        except Exception:
            report.corrupted_images.append(str(img_path))
            report.is_valid = False
            return

        report.valid_images += 1

        # 2. Verify Label File Existence
        if not label_path.exists():
            report.missing_labels.append(str(label_path))
            report.empty_labels.append(str(img_path))
            return

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            report.empty_labels.append(str(img_path))
            return

        # 3. Inspect Annotation Coordinates
        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                report.invalid_boxes.append({
                    "image": str(img_path),
                    "line": idx + 1,
                    "reason": "Fewer than 5 columns in YOLO text line",
                })
                report.is_valid = False
                continue

            try:
                cls_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                report.invalid_boxes.append({
                    "image": str(img_path),
                    "line": idx + 1,
                    "reason": "Non-numeric annotation value",
                })
                report.is_valid = False
                continue

            # Class ID Validation
            if cls_id < 0 or cls_id >= self.num_classes:
                report.invalid_class_ids.append({
                    "image": str(img_path),
                    "line": idx + 1,
                    "class_id": cls_id,
                    "max_expected": self.num_classes - 1,
                })
                report.is_valid = False

            # Zero Area Validation
            if w <= 0.0 or h <= 0.0:
                report.zero_area_boxes.append({
                    "image": str(img_path),
                    "line": idx + 1,
                    "width": w,
                    "height": h,
                })
                report.is_valid = False

            # Out of Bounds Validation (Normalized YOLO coords must be within [0, 1])
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0

            if x1 < -0.05 or y1 < -0.05 or x2 > 1.05 or y2 > 1.05:
                report.out_of_bounds_boxes.append({
                    "image": str(img_path),
                    "line": idx + 1,
                    "box": [x1, y1, x2, y2],
                })
                report.is_valid = False

    def validate_dataset(
        self,
        img_paths: List[Union[str, Path]],
        label_paths: List[Union[str, Path]],
    ) -> ValidationReport:
        """Validate an entire list of image and label file paths.

        Args:
            img_paths: List of image file paths.
            label_paths: List of corresponding label file paths.

        Returns:
            ValidationReport dataclass containing audit findings.
        """
        report = ValidationReport(total_images=len(img_paths))

        for img_p, lbl_p in zip(img_paths, label_paths):
            self.validate_yolo_pair(Path(img_p), Path(lbl_p), report)

        return report

    def save_report(
        self,
        report: ValidationReport,
        output_json: Union[str, Path] = "logs/dataset_validation_report.json",
    ) -> Path:
        """Serialize ValidationReport to JSON.

        Args:
            report: ValidationReport instance.
            output_json: Target JSON file path.

        Returns:
            Path to saved JSON file.
        """
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(report)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return out_path


__all__ = ["DatasetValidator", "ValidationReport"]
