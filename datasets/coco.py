"""COCO Format Object Detection Dataset Loader for Mhyra Vision.

Parses COCO JSON annotations (categories, images, annotations with [x, y, w, h] bounding boxes).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore

from datasets.base import BaseDetectionDataset
from datasets.transforms import Letterbox, Normalize


class COCODataset(BaseDetectionDataset):
    """Dataset loader for COCO JSON annotation format."""

    def __init__(
        self,
        annotation_json: Union[str, Path],
        images_dir: Optional[Union[str, Path]] = None,
        img_size: Tuple[int, int] = (640, 640),
        max_gt: int = 50,
        transforms: Optional[Any] = None,
        is_train: bool = True,
        img_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize COCODataset."""
        super().__init__(img_size=img_size, max_gt=max_gt, transforms=transforms, is_train=is_train)
        img_directory = images_dir if images_dir is not None else img_dir
        self.annotation_json = Path(annotation_json)
        self.images_dir = Path(img_directory) if img_directory else Path(".")

        self.images_info, self.image_anns, self.class_names = self._parse_coco_json(self.annotation_json)
        self.num_classes = len(self.class_names)
        self.img_ids = list(self.images_info.keys())

        self.letterbox = Letterbox(new_shape=img_size, auto=False)
        self.normalize = Normalize()

    def _parse_coco_json(
        self, json_path: Path
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]], List[str]]:
        """Parse COCO format JSON annotation file."""
        if not json_path.exists():
            return {}, {}, []

        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        cats = sorted(coco_data.get("categories", []), key=lambda c: c.get("id", 0))
        cat_id_to_index = {c.get("id"): idx for idx, c in enumerate(cats)} if cats else {}
        class_names = [c.get("name", str(c.get("id"))) for c in cats] if cats else ["object"]

        images_info = {img["id"]: img for img in coco_data.get("images", [])}

        image_anns: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in images_info}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in image_anns:
                cat_id = ann["category_id"]
                ann["cls_idx"] = cat_id_to_index.get(cat_id, cat_id)
                image_anns[img_id].append(ann)

        return images_info, image_anns, class_names

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(
        self, index: int
    ) -> Tuple[Any, Any, Any, Any]:
        """Fetch item at index."""
        img_id = self.img_ids[index]
        img_meta = self.images_info[img_id]
        img_filename = img_meta.get("file_name", img_meta.get("filename", ""))
        img_path = self.images_dir / img_filename

        img_orig = self.load_image(img_path)

        boxes_list: List[List[float]] = []
        labels_list: List[float] = []

        anns = self.image_anns.get(img_id, [])
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes_list.append([x, y, x + w, y + h])
            labels_list.append(ann["cls_idx"])

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


__all__ = ["COCODataset"]
