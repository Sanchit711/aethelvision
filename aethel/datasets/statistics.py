"""Dataset Statistics Analyzer for Aethel Vision.

Computes comprehensive dataset health metrics, object counts, class distributions,
aspect ratio stats, and bounding box area breakdowns. Output is fully JSON serializable.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from aethel.datasets.metadata import ImageInfo
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.statistics")


class DatasetStatistics:
    """Analyzer for computing quantitative statistics across a dataset."""

    @staticmethod
    def compute(samples: List[ImageInfo], class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute quantitative statistics dictionary from sample list.

        Args:
            samples: List of ImageInfo sample objects.
            class_names: Optional list of class name strings.

        Returns:
            JSON-serializable statistics summary dictionary.
        """
        total_images = len(samples)
        if total_images == 0:
            return {
                "total_images": 0,
                "total_annotations": 0,
                "empty_images": 0,
                "completeness_score": 0.0,
            }

        total_annotations = 0
        empty_images = 0
        class_counts: Dict[int, int] = {}
        objects_per_image: List[int] = []
        image_widths: List[int] = []
        image_heights: List[int] = []
        aspect_ratios: List[float] = []

        # Area scale breakdown (COCO definition: small < 32^2, medium < 96^2, large > 96^2)
        small_boxes = 0
        medium_boxes = 0
        large_boxes = 0

        for sample in samples:
            ann_count = len(sample.annotations)
            total_annotations += ann_count
            objects_per_image.append(ann_count)

            if ann_count == 0:
                empty_images += 1

            if sample.width > 0 and sample.height > 0:
                image_widths.append(sample.width)
                image_heights.append(sample.height)
                aspect_ratios.append(sample.width / sample.height)

            for ann in sample.annotations:
                cid = ann.class_id
                class_counts[cid] = class_counts.get(cid, 0) + 1

                # Calculate box area in pixel space
                w_px, h_px = 0.0, 0.0
                if ann.format == "xywh_norm" and sample.width > 0 and sample.height > 0:
                    w_px = ann.bbox[2] * sample.width
                    h_px = ann.bbox[3] * sample.height
                elif ann.format == "xyxy":
                    w_px = ann.bbox[2] - ann.bbox[0]
                    h_px = ann.bbox[3] - ann.bbox[1]
                elif ann.format == "xywh_abs":
                    w_px = ann.bbox[2]
                    h_px = ann.bbox[3]

                area = w_px * h_px
                if area < 32 * 32:
                    small_boxes += 1
                elif area < 96 * 96:
                    medium_boxes += 1
                else:
                    large_boxes += 1

        # Class distribution with names
        class_distribution: Dict[str, int] = {}
        for cid, count in sorted(class_counts.items()):
            label = class_names[cid] if class_names and 0 <= cid < len(class_names) else f"class_{cid}"
            class_distribution[label] = count

        avg_w = float(np.mean(image_widths)) if image_widths else 0.0
        avg_h = float(np.mean(image_heights)) if image_heights else 0.0
        min_w = int(np.min(image_widths)) if image_widths else 0
        max_w = int(np.max(image_widths)) if image_widths else 0
        min_h = int(np.min(image_heights)) if image_heights else 0
        max_h = int(np.max(image_heights)) if image_heights else 0

        completeness_score = float((total_images - empty_images) / total_images) if total_images > 0 else 0.0

        stats: Dict[str, Any] = {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "empty_images": empty_images,
            "completeness_score": round(completeness_score, 4),
            "objects_per_image": {
                "mean": round(float(np.mean(objects_per_image)), 2),
                "std": round(float(np.std(objects_per_image)), 2),
                "min": int(np.min(objects_per_image)),
                "max": int(np.max(objects_per_image)),
            },
            "image_dimensions": {
                "average_width": round(avg_w, 1),
                "average_height": round(avg_h, 1),
                "smallest_image": [min_w, min_h],
                "largest_image": [max_w, max_h],
                "average_aspect_ratio": round(float(np.mean(aspect_ratios)), 3) if aspect_ratios else 1.0,
            },
            "class_distribution": class_distribution,
            "bbox_scale_distribution": {
                "small_boxes_under_32px": small_boxes,
                "medium_boxes_32_to_96px": medium_boxes,
                "large_boxes_over_96px": large_boxes,
            },
        }

        return stats

    @staticmethod
    def save_json(stats: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Save statistics dictionary to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved dataset statistics report to {path}")


__all__ = ["DatasetStatistics"]
