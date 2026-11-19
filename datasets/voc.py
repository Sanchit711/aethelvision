"""Pascal VOC Format Object Detection Dataset Loader for Mhyra Vision.

Parses Pascal VOC XML annotations (<annotation><object><name>, <bndbox>...).
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore

from datasets.base import BaseDetectionDataset
from datasets.transforms import Letterbox, Normalize

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]


class VOCDataset(BaseDetectionDataset):
    """Dataset loader for Pascal VOC XML annotation format."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        class_names: Optional[List[str]] = None,
        img_size: Tuple[int, int] = (640, 640),
        max_gt: int = 50,
        transforms: Optional[Any] = None,
        is_train: bool = True,
        xml_paths: Optional[List[Any]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize VOCDataset."""
        super().__init__(img_size=img_size, max_gt=max_gt, transforms=transforms, is_train=is_train)
        if xml_paths is not None:
            self.data_dir = Path(".")
            self.xml_paths = [Path(p) for p in xml_paths]
            self.img_paths = [p.with_suffix(".jpg") for p in self.xml_paths]
            self.class_to_idx = class_to_idx if class_to_idx is not None else {c: i for i, c in enumerate(VOC_CLASSES)}
            self.class_names = list(self.class_to_idx.keys())
            self.num_classes = len(self.class_names)
        elif data_dir is not None:
            self.data_dir = Path(data_dir)
            self.class_names = class_names or VOC_CLASSES
            self.num_classes = len(self.class_names)
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.img_paths, self.xml_paths = self._discover_voc_files(self.data_dir, split)
        else:
            self.data_dir = Path(".")
            self.xml_paths = []
            self.img_paths = []
            self.class_to_idx = {}
            self.class_names = []
            self.num_classes = 0
        self.letterbox = Letterbox(new_shape=img_size, auto=False)
        self.normalize = Normalize()

    def _discover_voc_files(self, data_dir: Path, split: str) -> Tuple[List[Path], List[Path]]:
        """Discover image and XML annotation paths."""
        img_dir = data_dir / "JPEGImages"
        xml_dir = data_dir / "Annotations"

        if not img_dir.exists():
            img_dir = data_dir / "images"
        if not xml_dir.exists():
            xml_dir = data_dir / "labels"

        split_file = data_dir / "ImageSets" / "Main" / f"{split}.txt"
        img_ids: List[str] = []

        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                img_ids = [line.strip() for line in f if line.strip()]

        if img_ids:
            img_paths = [img_dir / f"{i}.jpg" for i in img_ids]
            xml_paths = [xml_dir / f"{i}.xml" for i in img_ids]
        else:
            xml_paths = sorted(list(xml_dir.glob("*.xml"))) if xml_dir.exists() else []
            img_paths = [img_dir / f"{p.stem}.jpg" for p in xml_paths]

        return img_paths, xml_paths

    def _parse_xml(self, xml_path: Path) -> Tuple[List[List[float]], List[float]]:
        """Parse ground truth boxes and class labels from XML file."""
        boxes = []
        labels = []
        if not xml_path.exists():
            return boxes, labels

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name")
            if name is None or name.text not in self.class_to_idx:
                continue

            cls_id = float(self.class_to_idx[name.text])
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = float(bndbox.find("xmin").text or 0.0)
                ymin = float(bndbox.find("ymin").text or 0.0)
                xmax = float(bndbox.find("xmax").text or 0.0)
                ymax = float(bndbox.find("ymax").text or 0.0)

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)

        return boxes, labels

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(
        self, index: int
    ) -> Tuple[Any, Any, Any, Any]:
        """Fetch item at index."""
        img_path = self.img_paths[index]
        xml_path = self.xml_paths[index]

        img_orig = self.load_image(img_path)
        boxes_list, labels_list = self._parse_xml(xml_path)

        boxes_arr = np.array(boxes_list, dtype=np.float32) if boxes_list else np.zeros((0, 4), dtype=np.float32)

        img_letter, boxes_letter = self.letterbox(img_orig, boxes_arr)
        if self.transforms is not None:
            img_letter, boxes_letter = self.transforms(img_letter, boxes_letter)

        img_tensor, _ = self.normalize(img_letter, boxes_letter)

        if torch is None:
            boxes_tensor = boxes_letter if boxes_letter is not None else np.zeros((0, 4), dtype=np.float32)
            labels_tensor = np.array(labels_list, dtype=np.float32) if labels_list else np.zeros((0, 1), dtype=np.float32)
        else:
            boxes_tensor = torch.from_numpy(boxes_letter).float() if boxes_letter is not None else torch.zeros((0, 4))
            labels_tensor = torch.tensor(labels_list, dtype=torch.float32) if labels_list else torch.zeros((0, 1))

        padded_labels, padded_boxes, mask_gt = self.pad_targets(boxes_tensor, labels_tensor)
        return img_tensor, padded_labels, padded_boxes, mask_gt


__all__ = ["VOCDataset", "VOC_CLASSES"]
