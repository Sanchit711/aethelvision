"""Multi-Format Dataset and Label Parsing Subsystem for Aethel Vision.

Supports YOLO TXT, Pascal VOC XML, COCO JSON, and automatic format detection.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from aethel.datasets.metadata import AnnotationInfo, ImageInfo
from aethel.utils.exceptions import DatasetException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.parser")


class LabelParser:
    """Parser for individual label files across supported formats."""

    @staticmethod
    def parse_yolo_file(
        label_path: Union[str, Path],
        img_width: int = 1,
        img_height: int = 1,
    ) -> List[AnnotationInfo]:
        """Parse YOLO text label file format [class_id, x_center, y_center, width, height].

        Args:
            label_path: Path to .txt label file.
            img_width: Image width (for coordinate validation).
            img_height: Image height (for coordinate validation).

        Returns:
            List of AnnotationInfo objects.
        """
        path = Path(label_path)
        if not path.exists():
            return []

        annotations = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        logger.warning(f"Skipping malformed line {line_idx} in {path.name}: expected 5+ columns, got {len(parts)}")
                        continue

                    class_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    annotations.append(
                        AnnotationInfo(
                            class_id=class_id,
                            bbox=(xc, yc, w, h),
                            format="xywh_norm",
                        )
                    )
            return annotations
        except Exception as e:
            raise DatasetException(
                message=f"Failed to parse YOLO label file: {path.name}",
                details=str(e),
            )

    @staticmethod
    def parse_voc_file(xml_path: Union[str, Path], class_to_id: Dict[str, int]) -> Tuple[ImageInfo, List[AnnotationInfo]]:
        """Parse Pascal VOC XML annotation file.

        Args:
            xml_path: Path to XML file.
            class_to_id: Mapping from class name string to integer ID.

        Returns:
            Tuple of (ImageInfo, List[AnnotationInfo]).
        """
        path = Path(xml_path)
        if not path.exists():
            raise DatasetException(message=f"VOC XML file not found: {path}")

        try:
            tree = ET.parse(path)
            root = tree.getroot()

            filename = root.findtext("filename", "")
            size_elem = root.find("size")
            width = int(size_elem.findtext("width", "0")) if size_elem is not None else 0
            height = int(size_elem.findtext("height", "0")) if size_elem is not None else 0

            annotations = []
            for obj in root.findall("object"):
                name = obj.findtext("name", "")
                if name not in class_to_id:
                    continue
                class_id = class_to_id[name]

                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin = float(bndbox.findtext("xmin", "0"))
                    ymin = float(bndbox.findtext("ymin", "0"))
                    xmax = float(bndbox.findtext("xmax", "0"))
                    ymax = float(bndbox.findtext("ymax", "0"))

                    annotations.append(
                        AnnotationInfo(
                            class_id=class_id,
                            bbox=(xmin, ymin, xmax, ymax),
                            format="xyxy",
                        )
                    )

            img_info = ImageInfo(
                image_path=path.with_name(filename) if filename else path.with_suffix(".jpg"),
                height=height,
                width=width,
                label_path=path,
                annotations=annotations,
            )
            return img_info, annotations
        except Exception as e:
            raise DatasetException(
                message=f"Failed to parse VOC XML file: {path.name}",
                details=str(e),
            )


class DatasetParser:
    """Orchestrator for parsing multi-format dataset directories."""

    @staticmethod
    def detect_format(dataset_dir: Union[str, Path]) -> str:
        """Automatically detect dataset format ('yolo', 'coco', 'voc').

        Args:
            dataset_dir: Path to dataset directory.

        Returns:
            Format string identifier.
        """
        dir_path = Path(dataset_dir)
        if (dir_path / "annotations.json").exists() or list(dir_path.glob("*.json")):
            return "coco"
        if list(dir_path.rglob("*.xml")):
            return "voc"
        if (dir_path / "labels").exists() or list(dir_path.rglob("*.txt")):
            return "yolo"
        return "yolo"

    @staticmethod
    def parse_coco_json(json_path: Union[str, Path]) -> Tuple[List[ImageInfo], Dict[int, str]]:
        """Parse COCO format JSON annotation file.

        Args:
            json_path: Path to COCO instances_*.json file.

        Returns:
            Tuple of (List[ImageInfo], Dict[category_id, category_name]).
        """
        path = Path(json_path)
        if not path.exists():
            raise DatasetException(message=f"COCO JSON file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

            # Map image_id to ImageInfo
            img_map: Dict[int, ImageInfo] = {}
            for img in data.get("images", []):
                img_id = img["id"]
                file_name = img["file_name"]
                img_map[img_id] = ImageInfo(
                    image_path=path.parent / file_name,
                    height=img.get("height", 0),
                    width=img.get("width", 0),
                )

            # Map annotations
            for ann in data.get("annotations", []):
                img_id = ann["image_id"]
                if img_id not in img_map:
                    continue

                cat_id = ann["category_id"]
                bbox = ann["bbox"]  # [x, y, w, h]
                is_crowd = bool(ann.get("iscrowd", 0))

                ann_info = AnnotationInfo(
                    class_id=cat_id,
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    format="xywh_abs",
                    area=float(ann.get("area", bbox[2] * bbox[3])),
                    is_crowd=is_crowd,
                )
                img_map[img_id].annotations.append(ann_info)

            return list(img_map.values()), categories
        except Exception as e:
            raise DatasetException(
                message=f"Failed to parse COCO JSON file: {path.name}",
                details=str(e),
            )


__all__ = ["LabelParser", "DatasetParser"]
