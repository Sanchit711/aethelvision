"""Dataset Quality Analytics and Reporting for Aethel Vision Framework.

Computes class distributions, bounding box statistics, aspect ratio distributions,
class imbalance ratios, and dataset quality scores. Exports JSON and HTML reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from aethel.utils.logger import get_logger

logger = get_logger("aethel.utils.dataset_reporter")


class DatasetReporter:
    """Analytics engine for computing dataset quality metrics and diagnostic reports."""

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        self.class_names = class_names or []

    def analyze(
        self,
        num_images: int,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze annotation records and compute diagnostic quality statistics.

        Args:
            num_images: Total image count in dataset.
            annotations: List of dicts containing ('class_id', 'bbox' [x1, y1, x2, y2]).

        Returns:
            Dictionary of dataset statistics.
        """
        total_objects = len(annotations)
        class_counts: Dict[int, int] = {}
        aspect_ratios: List[float] = []
        box_areas: List[float] = []

        for ann in annotations:
            cls_id = ann.get("class_id", 0)
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

            bbox = ann.get("bbox", [0, 0, 1, 1])
            w = max(bbox[2] - bbox[0], 1.0)
            h = max(bbox[3] - bbox[1], 1.0)
            aspect_ratios.append(w / h)
            box_areas.append(w * h)

        # Class distribution
        class_dist = {
            self.class_names[c] if c < len(self.class_names) else f"class_{c}": count
            for c, count in class_counts.items()
        }

        # Class imbalance calculation (max count / min count)
        counts_list = list(class_counts.values()) if class_counts else [1]
        imbalance_ratio = float(max(counts_list) / max(min(counts_list), 1))

        # Quality score heuristic (100 base penalty)
        quality_score = max(100.0 - min(imbalance_ratio * 2.0, 50.0), 50.0)

        report = {
            "num_images": num_images,
            "total_objects": total_objects,
            "avg_objects_per_image": round(total_objects / max(num_images, 1), 2),
            "class_distribution": class_dist,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "dataset_quality_score": round(quality_score, 1),
            "bbox_statistics": {
                "mean_area": round(float(np.mean(box_areas)), 2) if box_areas else 0.0,
                "median_aspect_ratio": round(float(np.median(aspect_ratios)), 2) if aspect_ratios else 1.0,
            },
        }

        logger.info(f"Analyzed dataset quality: Score = {quality_score:.1f}/100 | Images = {num_images} | Objects = {total_objects}")
        return report

    def save_json(self, report: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Save dataset quality report to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved dataset quality report to {path}")
        return path


__all__ = ["DatasetReporter"]
