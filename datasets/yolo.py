"""Production YOLO Format Object Detection Dataset Loader for Mhyra Vision.

Supports YOLO text annotation format:
- Auto-parses `data.yaml` (names as list `names: ['person']` or dict `names: {0: 'person'}`)
- Auto-discovers `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels`
- Performs real image loading and letterboxing / transforms
- Converts normalized [cls, xc, yc, w, h] to pixel [x1, y1, x2, y2]
- Returns tensors compatible with DetectorTrainer & CompositeDetectionLoss
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

try:
    import torch
except Exception:
    torch = None  # type: ignore

from datasets.base import BaseDetectionDataset
from datasets.transforms import Letterbox, Normalize


class YOLODataset(BaseDetectionDataset):
    """Production Dataset loader for YOLO text annotation format."""

    def __init__(
        self,
        data_path_or_yaml: Optional[Union[str, Path]] = None,
        split: str = "train",
        img_size: Tuple[int, int] = (640, 640),
        max_gt: int = 50,
        transforms: Optional[Any] = None,
        is_train: bool = True,
        img_paths: Optional[List[Any]] = None,
        label_paths: Optional[List[Any]] = None,
    ) -> None:
        """Initialize YOLODataset."""
        super().__init__(img_size=img_size, max_gt=max_gt, transforms=transforms, is_train=is_train)
        if img_paths is not None and label_paths is not None:
            self.root_path = Path(".")
            self.data_manifest = {}
            self.class_names = ["object"]
            self.num_classes = 1
            self.img_paths = [Path(p) for p in img_paths]
            self.label_paths = [Path(p) for p in label_paths]
        elif data_path_or_yaml is not None:
            self.root_path, self.data_manifest = self._parse_dataset_manifest(data_path_or_yaml)
            self.class_names = self._parse_class_names(self.data_manifest)
            self.num_classes = len(self.class_names)
            self.img_paths, self.label_paths = self._discover_split_files(self.root_path, self.data_manifest, split)
        else:
            self.root_path = Path(".")
            self.data_manifest = {}
            self.class_names = []
            self.num_classes = 0
            self.img_paths, self.label_paths = [], []

        self.letterbox = Letterbox(new_shape=img_size, auto=False)
        self.normalize = Normalize()

    def _parse_dataset_manifest(
        self, data_path_or_yaml: Union[str, Path]
    ) -> Tuple[Path, Dict[str, Any]]:
        """Parse root path and data.yaml configuration dictionary."""
        path = Path(data_path_or_yaml)
        manifest: Dict[str, Any] = {}

        if path.is_file() and path.suffix.lower() in (".yaml", ".yml"):
            root_dir = path.parent
            with open(path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f) or {}
        elif path.is_dir():
            root_dir = path
            yaml_candidates = [path / "data.yaml", path / "dataset.yaml", path / "data.yml"]
            for cand in yaml_candidates:
                if cand.exists():
                    with open(cand, "r", encoding="utf-8") as f:
                        manifest = yaml.safe_load(f) or {}
                    break
        else:
            root_dir = path

        if "path" in manifest and manifest["path"]:
            manifest_path = Path(manifest["path"])
            if manifest_path.is_absolute():
                root_dir = manifest_path
            else:
                root_dir = (root_dir / manifest_path).resolve()

        return root_dir, manifest

    def _parse_class_names(self, manifest: Dict[str, Any]) -> List[str]:
        """Extract ordered class names list from data manifest dictionary."""
        names = manifest.get("names", [])
        if isinstance(names, list):
            return [str(n) for n in names]
        elif isinstance(names, dict):
            sorted_keys = sorted([int(k) for k in names.keys()])
            return [str(names[k]) for k in sorted_keys]
        elif "nc" in manifest:
            return [f"class_{i}" for i in range(int(manifest["nc"]))]
        return ["class_0"]

    def _discover_split_files(
        self,
        root_dir: Path,
        manifest: Dict[str, Any],
        split: str,
    ) -> Tuple[List[Path], List[Path]]:
        """Discover matching image and text label files for split."""
        split_key = split.lower()
        if split_key in ("val", "validation"):
            split_dirs = ["valid", "val"]
        else:
            split_dirs = [split_key]

        rel_img_path = None
        for key in split_dirs:
            if key in manifest:
                rel_img_path = manifest[key]
                break

        img_dir = None
        if rel_img_path:
            cand = root_dir / rel_img_path
            if cand.exists():
                img_dir = cand

        if img_dir is None:
            for s_name in split_dirs:
                cand = root_dir / s_name / "images"
                if cand.exists():
                    img_dir = cand
                    break
                cand_sub = root_dir / s_name
                if cand_sub.exists() and any(cand_sub.glob("*.jpg")):
                    img_dir = cand_sub
                    break

        if img_dir is None:
            # Fallback to root_dir only if no split subdirectories exist at all in dataset
            has_split_subdirs = any((root_dir / d).exists() for d in ["train", "val", "valid", "test"])
            if not has_split_subdirs:
                img_dir = root_dir
            else:
                return [], []

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if img_dir.exists():
            img_paths = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
            if not img_paths:
                img_paths = sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts])
        else:
            img_paths = []

        label_paths = []
        for img_p in img_paths:
            parts = list(img_p.parts)
            if "images" in parts:
                idx = len(parts) - 1 - parts[::-1].index("images")
                parts[idx] = "labels"
                lbl_p = Path(*parts).with_suffix(".txt")
            else:
                lbl_p = img_p.with_suffix(".txt")

            label_paths.append(lbl_p)

        return img_paths, label_paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(
        self, index: int
    ) -> Tuple[Any, Any, Any, Any]:
        """Fetch item at index."""
        img_path = self.img_paths[index]
        lbl_path = self.label_paths[index]

        img_orig = self.load_image(img_path)
        h_orig, w_orig = img_orig.shape[:2]

        boxes_list: List[List[float]] = []
        labels_list: List[float] = []

        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = float(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])

                    x1 = (xc - w / 2.0) * w_orig
                    y1 = (yc - h / 2.0) * h_orig
                    x2 = (xc + w / 2.0) * w_orig
                    y2 = (yc + h / 2.0) * h_orig

                    boxes_list.append([x1, y1, x2, y2])
                    labels_list.append(cls_id)

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


__all__ = ["YOLODataset"]
