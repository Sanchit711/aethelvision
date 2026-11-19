"""Production Dataset Statistics Analyzer for Mhyra Detector.

Computes annotation counts, class distributions, object size breakdowns, aspect ratios,
label density, and spatial center heatmaps for object detection datasets.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np


@dataclass
class StatisticsReport:
    """Dataset statistics metrics manifest."""

    total_images: int = 0
    total_annotations: int = 0
    label_density: float = 0.0
    class_distribution: Dict[int, int] = field(default_factory=dict)
    images_per_class: Dict[int, int] = field(default_factory=dict)
    object_size_distribution: Dict[str, int] = field(default_factory=lambda: {"small": 0, "medium": 0, "large": 0})
    aspect_ratio_distribution: Dict[str, int] = field(
        default_factory=lambda: {"ratio_<0.5": 0, "ratio_0.5_1.0": 0, "ratio_1.0_2.0": 0, "ratio_>2.0": 0}
    )
    spatial_center_heatmap: List[List[int]] = field(default_factory=lambda: [[0] * 10 for _ in range(10)])


class DatasetStatistics:
    """Computes comprehensive dataset analytics and spatial heatmaps."""

    @staticmethod
    def compute_from_yolo(
        label_paths: List[Union[str, Path]],
        img_size: Tuple[int, int] = (640, 640),
    ) -> StatisticsReport:
        """Compute statistics metrics from a list of YOLO annotation text files.

        Args:
            label_paths: List of YOLO text annotation file paths.
            img_size: Image (height, width) spatial dimensions.

        Returns:
            StatisticsReport populated with calculated metrics.
        """
        h_img, w_img = img_size
        report = StatisticsReport(total_images=len(label_paths))

        heatmap = np.zeros((10, 10), dtype=int)
        class_counts: Dict[int, int] = {}
        class_img_sets: Dict[int, set] = {}

        for img_idx, lbl_p in enumerate(label_paths):
            path = Path(lbl_p)
            if not path.exists():
                continue

            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    cls_id = int(float(parts[0]))
                    xc, yc, w_norm, h_norm = map(float, parts[1:5])
                except ValueError:
                    continue

                report.total_annotations += 1
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

                if cls_id not in class_img_sets:
                    class_img_sets[cls_id] = set()
                class_img_sets[cls_id].add(img_idx)

                # Absolute spatial size calculations
                w_px = w_norm * w_img
                h_px = h_norm * h_img
                area_px = w_px * h_px

                # Size distribution (COCO criteria: small <32^2, medium 32^2-96^2, large >96^2)
                if area_px < 1024:
                    report.object_size_distribution["small"] += 1
                elif area_px <= 9216:
                    report.object_size_distribution["medium"] += 1
                else:
                    report.object_size_distribution["large"] += 1

                # Aspect ratio distribution
                ar = w_px / max(h_px, 1e-4)
                if ar < 0.5:
                    report.aspect_ratio_distribution["ratio_<0.5"] += 1
                elif ar <= 1.0:
                    report.aspect_ratio_distribution["ratio_0.5_1.0"] += 1
                elif ar <= 2.0:
                    report.aspect_ratio_distribution["ratio_1.0_2.0"] += 1
                else:
                    report.aspect_ratio_distribution["ratio_>2.0"] += 1

                # Spatial center heatmap (10x10 grid)
                grid_x = min(int(xc * 10), 9)
                grid_y = min(int(yc * 10), 9)
                heatmap[grid_y, grid_x] += 1

        report.label_density = report.total_annotations / max(report.total_images, 1)
        report.class_distribution = class_counts
        report.images_per_class = {k: len(v) for k, v in class_img_sets.items()}
        report.spatial_center_heatmap = heatmap.tolist()

        return report

    @staticmethod
    def save_report(
        report: StatisticsReport,
        output_json: Union[str, Path] = "logs/dataset_statistics.json",
    ) -> Path:
        """Serialize StatisticsReport to JSON file.

        Args:
            report: StatisticsReport instance.
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


__all__ = ["DatasetStatistics", "StatisticsReport"]
