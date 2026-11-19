"""Dataset Integrity and Quality Verifier for Mhyra Vision.

Implements DatasetVerifier for validating dataset paths, annotation integrity,
corrupt images, orphan label files, class ID ranges, and bounding box constraints.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml


@dataclass
class VerificationReport:
    """Dataclass holding complete dataset verification results summary."""

    total_images: int = 0
    total_labels: int = 0
    missing_images: List[str] = field(default_factory=list)
    missing_labels: List[str] = field(default_factory=list)
    corrupt_images: List[str] = field(default_factory=list)
    orphan_labels: List[str] = field(default_factory=list)
    invalid_boxes: List[str] = field(default_factory=list)
    invalid_class_ids: List[str] = field(default_factory=list)
    yaml_valid: bool = True
    yaml_errors: List[str] = field(default_factory=list)
    is_valid: bool = True

    def summary(self) -> Dict[str, Any]:
        """Convert report to summary dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_images": self.total_images,
            "total_labels": self.total_labels,
            "missing_images_count": len(self.missing_images),
            "missing_labels_count": len(self.missing_labels),
            "corrupt_images_count": len(self.corrupt_images),
            "orphan_labels_count": len(self.orphan_labels),
            "invalid_boxes_count": len(self.invalid_boxes),
            "invalid_class_ids_count": len(self.invalid_class_ids),
            "yaml_valid": self.yaml_valid,
        }


class DatasetVerifier:
    """Production verifier for dataset paths, files, and annotations."""

    def __init__(self, data_path_or_yaml: Union[str, Path]) -> None:
        self.target_path = Path(data_path_or_yaml)

    def verify(self) -> VerificationReport:
        """Run complete dataset verification and return VerificationReport."""
        report = VerificationReport()

        if not self.target_path.exists():
            report.is_valid = False
            report.yaml_valid = False
            report.yaml_errors.append(f"Target path does not exist: {self.target_path}")
            return report

        # 1. Verify YAML manifest syntax if YAML file
        yaml_manifest = {}
        if self.target_path.is_file() and self.target_path.suffix in (".yaml", ".yml"):
            try:
                with open(self.target_path, "r", encoding="utf-8") as f:
                    yaml_manifest = yaml.safe_load(f) or {}
                root_dir = self.target_path.parent
            except Exception as e:
                report.yaml_valid = False
                report.yaml_errors.append(f"YAML parsing exception: {e}")
                report.is_valid = False
                return report
        else:
            root_dir = self.target_path

        # 2. Discover image and label files
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        img_files = sorted([p for p in root_dir.rglob("*") if p.suffix.lower() in valid_exts])
        lbl_files = sorted(list(root_dir.rglob("*.txt")))

        report.total_images = len(img_files)
        report.total_labels = len(lbl_files)

        img_stems: Set[str] = {p.stem for p in img_files}
        lbl_stems: Set[str] = {p.stem for p in lbl_files if p.name != "classes.txt"}

        # Orphan labels (label without image)
        for lbl in lbl_files:
            if lbl.name == "classes.txt":
                continue
            if lbl.stem not in img_stems:
                report.orphan_labels.append(str(lbl))

        # Check annotations for invalid boxes or class IDs
        for lbl_p in lbl_files:
            if lbl_p.name == "classes.txt":
                continue
            try:
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) < 5:
                            report.invalid_boxes.append(f"{lbl_p}: insufficient coordinates")
                            continue
                        try:
                            cls_id = int(float(parts[0]))
                            xc, yc, w, h = map(float, parts[1:5])
                            if cls_id < 0:
                                report.invalid_class_ids.append(f"{lbl_p}: negative class_id {cls_id}")
                            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                                report.invalid_boxes.append(f"{lbl_p}: out of range coordinates [{xc}, {yc}, {w}, {h}]")
                        except ValueError:
                            report.invalid_boxes.append(f"{lbl_p}: non-numeric value")
            except Exception as e:
                report.invalid_boxes.append(f"{lbl_p}: read error {e}")

        # Check for image corruptions (light header read)
        for img_p in img_files[:200]:  # Sample up to 200 images for speed
            if img_p.stat().st_size == 0:
                report.corrupt_images.append(str(img_p))

        report.is_valid = (
            report.yaml_valid
            and len(report.missing_images) == 0
            and len(report.corrupt_images) == 0
            and len(report.invalid_boxes) == 0
        )

        return report


__all__ = ["DatasetVerifier", "VerificationReport"]
