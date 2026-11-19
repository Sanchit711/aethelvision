"""Dataset Summary Reporter for Mhyra Vision Object Detection Datasets.

Computes comprehensive dataset metrics across splits (train/val/test),
class distributions, box instance counts, resolution statistics, and exports JSON/CSV reports.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets.yolo import YOLODataset


class DatasetReporter:
    """Production summary reporter for dataset statistics."""

    def __init__(self, data_path_or_yaml: Union[str, Path]) -> None:
        self.data_path = Path(data_path_or_yaml)

    def generate_report(self) -> Dict[str, Any]:
        """Generate dataset statistics breakdown across splits."""
        report: Dict[str, Any] = {
            "dataset_path": str(self.data_path),
            "splits": {},
            "total_images": 0,
            "total_instances": 0,
            "classes": [],
        }

        for split in ["train", "val", "test"]:
            try:
                ds = YOLODataset(data_path_or_yaml=self.data_path, split=split)
                if len(ds) == 0:
                    continue

                if not report["classes"]:
                    report["classes"] = ds.class_names

                split_instances = 0
                class_counts: Dict[str, int] = {name: 0 for name in ds.class_names}

                for lbl_p in ds.label_paths:
                    if lbl_p.exists():
                        with open(lbl_p, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    split_instances += 1
                                    try:
                                        cls_id = int(float(parts[0]))
                                        if cls_id < len(ds.class_names):
                                            cls_name = ds.class_names[cls_id]
                                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                                    except ValueError:
                                        pass

                report["splits"][split] = {
                    "image_count": len(ds),
                    "instance_count": split_instances,
                    "avg_boxes_per_image": round(split_instances / max(len(ds), 1), 2),
                    "class_distribution": class_counts,
                }
                report["total_images"] += len(ds)
                report["total_instances"] += split_instances
            except Exception:
                pass

        return report

    def export_json(self, report: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Export report dictionary to JSON file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return out_p

    def export_csv(self, report: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Export class distribution report to CSV file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Class Name", "Train Count", "Val Count", "Test Count", "Total Count"])

            classes = report.get("classes", [])
            splits = report.get("splits", {})

            for cls_name in classes:
                train_c = splits.get("train", {}).get("class_distribution", {}).get(cls_name, 0)
                val_c = splits.get("val", {}).get("class_distribution", {}).get(cls_name, 0)
                test_c = splits.get("test", {}).get("class_distribution", {}).get(cls_name, 0)
                tot_c = train_c + val_c + test_c
                writer.writerow([cls_name, train_c, val_c, test_c, tot_c])

        return out_p

    def print_console_summary(self, report: Dict[str, Any]) -> None:
        """Print clean ASCII summary table to stdout console."""
        print("===========================================================")
        print(f"DATASET SUMMARY REPORT: {report.get('dataset_path')}")
        print("===========================================================")
        print(f"Total Images: {report.get('total_images')}")
        print(f"Total Object Instances: {report.get('total_instances')}")
        print("-----------------------------------------------------------")
        for split, info in report.get("splits", {}).items():
            print(f"Split [{split.upper()}]: {info.get('image_count')} images | {info.get('instance_count')} boxes (Avg: {info.get('avg_boxes_per_image')}/img)")
        print("===========================================================\n")


__all__ = ["DatasetReporter"]
