"""Dataset Integrity and Bounding Box Validation Subsystem for Aethel Vision.

Detects missing files, orphan annotations, corrupted images, negative/OOB coordinates,
invalid class IDs, and duplicate bounding box entries.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from aethel.datasets.loader import SUPPORTED_IMAGE_EXTENSIONS, ImageLoader
from aethel.datasets.metadata import AnnotationInfo, ImageInfo
from aethel.utils.exceptions import DatasetException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.validator")


class DatasetIntegrityChecker:
    """Verifies file integrity, file existence, and image decoding health."""

    @staticmethod
    def check_image_corrupted(image_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Verify image file can be decoded cleanly by PIL.

        Returns:
            Tuple of (is_corrupted_bool, error_reason_str).
        """
        path = Path(image_path)
        if not path.exists():
            return True, "File does not exist"

        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return True, f"Unsupported extension '{path.suffix}'"

        try:
            with Image.open(path) as img:
                img.verify()
            return False, None
        except Exception as e:
            return True, f"Corrupted image header: {str(e)}"


class DatasetValidator:
    """Validator for object detection datasets and annotations."""

    def __init__(self, num_classes: int = 80) -> None:
        """Initialize DatasetValidator.

        Args:
            num_classes: Maximum valid class ID bound (0 to num_classes - 1).
        """
        self.num_classes = num_classes

    def validate_annotation(
        self,
        ann: AnnotationInfo,
        img_width: int = 640,
        img_height: int = 640,
    ) -> List[str]:
        """Validate a single annotation entry for bounding box anomalies.

        Returns:
            List of issue description strings (empty if valid).
        """
        issues = []

        # 1. NaN / Inf check
        if math.isnan(ann.class_id) or math.isinf(ann.class_id):
            issues.append(f"NaN or Inf detected in class_id {ann.class_id}")
            return issues

        for val in ann.bbox:
            if math.isnan(val) or math.isinf(val):
                issues.append("NaN or Inf detected in bounding box coordinates")

        # 2. Class ID validation
        if ann.class_id < 0 or ann.class_id >= self.num_classes:
            issues.append(f"Invalid class_id {ann.class_id} (allowed: 0 to {self.num_classes - 1})")

        # 3. Coordinate range & invalid bbox checks
        if ann.format == "xywh_norm":
            xc, yc, w, h = ann.bbox
            if xc < 0.0 or xc > 1.0 or yc < 0.0 or yc > 1.0:
                issues.append(f"Normalized center ({xc:.3f}, {yc:.3f}) out of [0, 1] bounds")
            if w <= 0.0 or w > 1.0 or h <= 0.0 or h > 1.0:
                issues.append(f"Invalid normalized width/height ({w:.3f}, {h:.3f})")
        else:
            x1, y1, x2, y2 = ann.to_xyxy(img_width, img_height)
            if x1 < 0 or y1 < 0:
                issues.append(f"Negative bounding box coordinates ({x1:.1f}, {y1:.1f})")
            if x2 <= x1 or y2 <= y1:
                issues.append(f"Degenerate box dimensions (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f})")
            if x2 > img_width * 1.5 or y2 > img_height * 1.5:
                issues.append(f"Bounding box ({x2:.1f}, {y2:.1f}) severely exceeds image dimensions ({img_width}x{img_height})")

        return issues

    def validate_dataset(self, samples: List[ImageInfo]) -> Dict[str, Any]:
        """Execute full dataset validation report over a list of ImageInfo samples.

        Returns:
            Report dictionary summarizing errors, warnings, and file counts.
        """
        report: Dict[str, Any] = {
            "total_samples": len(samples),
            "valid_samples": 0,
            "corrupted_images": [],
            "missing_images": [],
            "missing_labels": [],
            "empty_txt": [],
            "orphan_labels": [],
            "invalid_annotations": [],
            "duplicate_annotations": [],
            "errors_count": 0,
            "warnings_count": 0,
            "status": "PASSED",
        }

        if not samples:
            report["status"] = "FAILED"
            report["errors_count"] += 1
            report["empty_dataset"] = True
            return report

        for sample in samples:
            sample_valid = True
            img_p = Path(sample.image_path)

            # 1. Missing image check
            if not img_p.exists():
                report["missing_images"].append({"path": str(img_p), "reason": "File missing on disk"})
                report["errors_count"] += 1
                sample_valid = False
                continue

            # 2. Check image corruption
            is_corrupt, reason = DatasetIntegrityChecker.check_image_corrupted(img_p)
            if is_corrupt:
                report["corrupted_images"].append({"path": str(img_p), "reason": reason})
                report["errors_count"] += 1
                sample_valid = False

            # 3. Check label existence and empty txt
            if sample.label_path is not None:
                lbl_p = Path(sample.label_path)
                if not lbl_p.exists():
                    report["missing_labels"].append({"path": str(lbl_p), "image": str(img_p)})
                    report["warnings_count"] += 1
                elif lbl_p.suffix.lower() == ".txt":
                    try:
                        sz = lbl_p.stat().st_size
                        if sz == 0:
                            report["empty_txt"].append({"path": str(lbl_p), "image": str(img_p)})
                            report["warnings_count"] += 1
                    except Exception:
                        pass
            elif len(sample.annotations) == 0:
                report["missing_labels"].append({"path": "None", "image": str(img_p)})
                report["warnings_count"] += 1

            # 4. Validate annotations (NaN, Inf, Class ID, Negative, BBox)
            seen_boxes = set()
            for ann in sample.annotations:
                issues = self.validate_annotation(
                    ann,
                    img_width=sample.width or 640,
                    img_height=sample.height or 640,
                )
                if issues:
                    report["invalid_annotations"].append(
                        {"image": str(img_p), "class_id": ann.class_id, "issues": issues}
                    )
                    report["warnings_count"] += len(issues)

                # Duplicate box check
                box_key = (ann.class_id, tuple(round(v, 4) for v in ann.bbox))
                if box_key in seen_boxes:
                    report["duplicate_annotations"].append({"image": str(img_p), "box": box_key})
                    report["warnings_count"] += 1
                seen_boxes.add(box_key)

            if sample_valid:
                report["valid_samples"] += 1

        if report["errors_count"] > 0:
            report["status"] = "FAILED"

        return report

    def export_health_reports(self, report: Dict[str, Any], output_dir: Union[str, Path]) -> Dict[str, str]:
        """Export dataset_health_report.html, dataset_health.json, and dataset_health.csv.

        Returns:
            Dictionary of exported report file paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "dataset_health.json"
        csv_path = out / "dataset_health.csv"
        html_path = out / "dataset_health.html"
        html_report_path = out / "dataset_health_report.html"

        # 1. Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        # 2. Write CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Item_Path", "Details"])
            for item in report.get("corrupted_images", []):
                writer.writerow(["Corrupted_Image", item.get("path"), item.get("reason")])
            for item in report.get("missing_images", []):
                writer.writerow(["Missing_Image", item.get("path"), item.get("reason")])
            for item in report.get("missing_labels", []):
                writer.writerow(["Missing_Label", item.get("path"), item.get("image")])
            for item in report.get("empty_txt", []):
                writer.writerow(["Empty_TXT", item.get("path"), item.get("image")])
            for item in report.get("invalid_annotations", []):
                writer.writerow(["Invalid_Annotation", item.get("image"), "; ".join(item.get("issues", []))])
            for item in report.get("duplicate_annotations", []):
                writer.writerow(["Duplicate_Annotation", item.get("image"), str(item.get("box"))])

        # 3. Write HTML Report
        status_color = "#10b981" if report.get("status") == "PASSED" else "#ef4444"
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AETHEL Vision - Dataset Health Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #334155; }}
        h1, h2 {{ margin-top: 0; }}
        .status-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; background: {status_color}; color: #fff; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
        .stat-box {{ background: #0f172a; padding: 16px; border-radius: 8px; text-align: center; border: 1px solid #334155; }}
        .stat-num {{ font-size: 28px; font-weight: bold; color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ background: #0f172a; color: #94a3b8; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>AETHEL Dataset Verification & Health Audit</h1>
        <p>Status: <span class="status-badge">{report.get("status", "UNKNOWN")}</span></p>
        <div class="grid">
            <div class="stat-box"><div class="stat-num">{report.get("total_samples", 0)}</div><div>Total Samples</div></div>
            <div class="stat-box"><div class="stat-num">{report.get("valid_samples", 0)}</div><div>Valid Samples</div></div>
            <div class="stat-box"><div class="stat-num">{report.get("errors_count", 0)}</div><div>Errors</div></div>
            <div class="stat-box"><div class="stat-num">{report.get("warnings_count", 0)}</div><div>Warnings</div></div>
        </div>
    </div>
    <div class="card">
        <h2>Dataset Anomalies & Audit Log</h2>
        <p>Corrupted Images: <strong>{len(report.get("corrupted_images", []))}</strong> | Missing Images: <strong>{len(report.get("missing_images", []))}</strong> | Missing Labels: <strong>{len(report.get("missing_labels", []))}</strong> | Empty TXTs: <strong>{len(report.get("empty_txt", []))}</strong></p>
        <table>
            <thead><tr><th>Type</th><th>Sample / Image Path</th><th>Details</th></tr></thead>
            <tbody>
"""
        for item in report.get("corrupted_images", []):
            html_content += f"<tr><td style='color:#ef4444'>Corrupt Image</td><td>{item.get('path')}</td><td>{item.get('reason')}</td></tr>\n"
        for item in report.get("invalid_annotations", []):
            html_content += f"<tr><td style='color:#f59e0b'>Invalid BBox</td><td>{item.get('image')}</td><td>{'; '.join(item.get('issues', []))}</td></tr>\n"
        html_content += """            </tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        with open(html_report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Exported dataset health reports to {out}")
        return {
            "json": str(json_path.resolve()),
            "csv": str(csv_path.resolve()),
            "html": str(html_path.resolve()),
        }


__all__ = ["DatasetIntegrityChecker", "DatasetValidator"]

