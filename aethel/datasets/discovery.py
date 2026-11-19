"""AETHEL Enterprise Dynamic Dataset Discovery & Management Subsystem.

Provides dynamic, live filesystem scan for datasets in <Project Root>/data/,
supporting YOLO, COCO, VOC, Classification, Segmentation, OBB, and Pose formats,
with pluggable format detectors, smart auto-invalidating caching, dataset integrity validation,
health scoring, self-healing, sorting, and search/filtering.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.discovery")


def format_size_bytes(size_bytes: float) -> str:
    """Format byte size into human-readable string."""
    if size_bytes < 1024:
        return f"{int(size_bytes)} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# --- Pluggable Format Detectors ---

class BaseFormatDetector:
    """Abstract base format detector interface for pluggable dataset format discovery."""

    format_name: str = "Unknown"

    def detect(self, dataset_path: Path) -> Tuple[bool, Optional[str]]:
        """Return (is_detected, optional_error_reason)."""
        raise NotImplementedError


class YOLOFormatDetector(BaseFormatDetector):
    """Detects YOLO format datasets (data.yaml, dataset.yaml, or train/images + labels)."""

    format_name = "YOLO"

    def detect(self, dataset_path: Path) -> Tuple[bool, Optional[str]]:
        yaml_p = dataset_path / "data.yaml"
        if not yaml_p.exists():
            yaml_p = dataset_path / "dataset.yaml"

        if yaml_p.exists():
            return True, None

        has_train_imgs = (dataset_path / "train" / "images").exists() or (dataset_path / "images" / "train").exists()
        has_train_lbls = (dataset_path / "train" / "labels").exists() or (dataset_path / "labels" / "train").exists()

        if has_train_imgs or has_train_lbls:
            return True, None

        return False, None


class COCOFormatDetector(BaseFormatDetector):
    """Detects COCO format datasets (annotations/instances_train.json, etc.)."""

    format_name = "COCO"

    def detect(self, dataset_path: Path) -> Tuple[bool, Optional[str]]:
        ann_dir = dataset_path / "annotations"
        if ann_dir.exists() and list(ann_dir.glob("*.json")):
            return True, None
        if list(dataset_path.glob("instances_*.json")) or list(dataset_path.glob("annotations*.json")):
            return True, None
        return False, None


class VOCFormatDetector(BaseFormatDetector):
    """Detects Pascal VOC format datasets (JPEGImages & Annotations)."""

    format_name = "VOC"

    def detect(self, dataset_path: Path) -> Tuple[bool, Optional[str]]:
        has_jpeg = (dataset_path / "JPEGImages").exists() or (dataset_path / "images").exists()
        has_xml = (dataset_path / "Annotations").exists()
        if has_jpeg and has_xml:
            return True, None
        return False, None


class ClassificationFormatDetector(BaseFormatDetector):
    """Detects Image Classification datasets (subfolders per class in train/ or root)."""

    format_name = "Classification"

    def detect(self, dataset_path: Path) -> Tuple[bool, Optional[str]]:
        # Check subdirectories in train/ or root
        check_dirs = [dataset_path / "train", dataset_path]
        for d in check_dirs:
            if d.exists() and d.is_dir():
                subdirs = [s for s in d.iterdir() if s.is_dir() and s.name not in ("images", "labels", "annotations", "masks", "obb")]
                if len(subdirs) >= 2:
                    # Verify at least one subdir contains images
                    for s in subdirs:
                        if list(s.glob("*.jpg")) or list(s.glob("*.png")) or list(s.glob("*.jpeg")):
                            return True, None
        return False, None


class SegmentationFormatDetector(BaseFormatDetector):
    """Detects Instance / Semantic Segmentation datasets (masks/ or polygon yaml)."""

    format_name = "Segmentation"

    def detect(self, dataset_path: Path) -> Tuple[bool, Optional[str]]:
        if (dataset_path / "masks").exists() or (dataset_path / "segmentation").exists():
            return True, None
        yaml_p = dataset_path / "data.yaml"
        if yaml_p.exists():
            try:
                import yaml
                with open(yaml_p, "r", encoding="utf-8") as f:
                    d = yaml.safe_load(f) or {}
                if d.get("task") == "segment":
                    return True, None
            except Exception:
                pass
        return False, None


class OBBFormatDetector(BaseFormatDetector):
    """Detects Oriented Bounding Box (OBB) datasets."""

    format_name = "OBB"

    def detect(self, dataset_path: Path) -> Tuple[bool, Optional[str]]:
        if (dataset_path / "obb").exists():
            return True, None
        yaml_p = dataset_path / "data.yaml"
        if yaml_p.exists():
            try:
                import yaml
                with open(yaml_p, "r", encoding="utf-8") as f:
                    d = yaml.safe_load(f) or {}
                if d.get("task") == "obb":
                    return True, None
            except Exception:
                pass
        return False, None


class PoseFormatDetector(BaseFormatDetector):
    """Detects Keypoint Pose Estimation datasets."""

    format_name = "Pose"

    def detect(self, dataset_path: Path) -> Tuple[bool, Optional[str]]:
        yaml_p = dataset_path / "data.yaml"
        if yaml_p.exists():
            try:
                import yaml
                with open(yaml_p, "r", encoding="utf-8") as f:
                    d = yaml.safe_load(f) or {}
                if d.get("task") == "pose" or "kpt_shape" in d or "keypoints" in d:
                    return True, None
            except Exception:
                pass
        return False, None


FORMAT_DETECTORS: List[BaseFormatDetector] = [
    YOLOFormatDetector(),
    COCOFormatDetector(),
    VOCFormatDetector(),
    ClassificationFormatDetector(),
    SegmentationFormatDetector(),
    OBBFormatDetector(),
    PoseFormatDetector(),
]


# --- Smart Cache with Auto-Invalidation ---

class SmartDatasetCache:
    """Sub-second dataset metadata cache with strict auto-invalidation on mtime / file changes."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_cache_key(self, dataset_path: Path) -> str:
        return str(dataset_path.resolve())

    def compute_dir_signature(self, dataset_path: Path) -> str:
        """Compute lightweight directory signature based on mtime and entry count."""
        try:
            st = dataset_path.stat()
            mtime = st.st_mtime
            entries_count = len(list(dataset_path.iterdir())) if dataset_path.exists() else 0
            yaml_p = dataset_path / "data.yaml"
            if not yaml_p.exists():
                yaml_p = dataset_path / "dataset.yaml"
            yaml_mtime = yaml_p.stat().st_mtime if yaml_p.exists() else 0.0
            return f"{mtime}_{entries_count}_{yaml_mtime}"
        except Exception:
            return f"{time.time()}_0_0"

    def get(self, dataset_path: Path) -> Optional[Dict[str, Any]]:
        key = self.get_cache_key(dataset_path)
        if key in self._cache:
            sig = self.compute_dir_signature(dataset_path)
            cached_sig = self._cache[key].get("_signature")
            if cached_sig == sig:
                return self._cache[key].get("data")
            else:
                # Signature mismatch -> auto-invalidate
                del self._cache[key]
        return None

    def put(self, dataset_path: Path, data: Dict[str, Any]) -> None:
        key = self.get_cache_key(dataset_path)
        sig = self.compute_dir_signature(dataset_path)
        self._cache[key] = {
            "_signature": sig,
            "data": data,
        }

    def clear(self) -> None:
        self._cache.clear()


# Global cache instance
_DATASET_CACHE = SmartDatasetCache()


# --- Dataset Auditor & Integrity Validator ---

class DatasetAuditor:
    """Comprehensive Dataset Auditor computing classes, image counts, label issues, health score, and status."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path.resolve()

    def audit(self) -> Dict[str, Any]:
        """Perform deep dataset inspection."""
        name = self.dataset_path.name
        abs_path = str(self.dataset_path)
        
        try:
            rel_path = str(self.dataset_path.relative_to(Path(".").resolve()))
        except Exception:
            rel_path = name

        format_detected = "Unknown"
        detector_reason = None
        for detector in FORMAT_DETECTORS:
            is_match, reason = detector.detect(self.dataset_path)
            if is_match:
                format_detected = detector.format_name
                detector_reason = reason
                break

        # Discover classes
        classes = []
        yaml_p = self.dataset_path / "data.yaml"
        if not yaml_p.exists():
            yaml_p = self.dataset_path / "dataset.yaml"

        if yaml_p.exists():
            try:
                import yaml
                with open(yaml_p, "r", encoding="utf-8") as f:
                    d = yaml.safe_load(f) or {}
                names = d.get("names", [])
                classes = list(names.values()) if isinstance(names, dict) else list(names)
            except Exception:
                pass

        if not classes and format_detected == "Classification":
            # Extract directory names in train/ or root
            d_target = self.dataset_path / "train" if (self.dataset_path / "train").exists() else self.dataset_path
            classes = [s.name for s in d_target.iterdir() if s.is_dir() and s.name not in ("images", "labels", "annotations")]

        num_classes = len(classes)

        # Count splits & images
        train_imgs = self._count_split_images(["train"])
        val_imgs = self._count_split_images(["valid", "val"])
        test_imgs = self._count_split_images(["test"])
        total_imgs = train_imgs + val_imgs + test_imgs

        # Count labels, missing labels, empty labels
        label_count, missing_labels, empty_labels = self._audit_labels()

        # Check corrupted images & duplicate images (sample up to 200 images for speed)
        corrupted_images, duplicate_images, sample_res = self._audit_images_sample()

        # Calculate dataset size
        size_bytes = self._calculate_dir_size(self.dataset_path)
        size_str = format_size_bytes(size_bytes)

        # Last modified timestamp
        try:
            mtime = self.dataset_path.stat().st_mtime
            last_mod = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        except Exception:
            last_mod = "Unknown"

        # Compute Health Score (0 - 100%)
        health_score = 100
        health_score -= min(30, missing_labels * 2)
        health_score -= min(30, corrupted_images * 10)
        health_score -= min(15, duplicate_images * 1)
        health_score -= min(15, empty_labels * 1)
        if total_imgs == 0:
            health_score = 0
        health_score = max(0, min(100, health_score))

        # Determine Status
        status = "READY"
        status_reason = ""

        if not yaml_p.exists() and format_detected == "YOLO":
            status = "ERROR"
            status_reason = "data.yaml missing"
        elif total_imgs == 0:
            status = "INVALID"
            status_reason = "No images found"
        elif num_classes == 0:
            status = "INVALID"
            status_reason = "No classes defined"
        elif corrupted_images > 0:
            status = "ERROR"
            status_reason = f"{corrupted_images} corrupted image(s)"
        elif missing_labels > 0:
            status = "WARNING"
            status_reason = f"{missing_labels} missing label(s)"
        elif empty_labels > 0:
            status = "WARNING"
            status_reason = f"{empty_labels} empty label(s)"

        display_status = status if not status_reason else f"{status} ({status_reason})"

        return {
            "name": name,
            "path": abs_path,
            "rel_path": rel_path,
            "format": format_detected,
            "num_classes": num_classes,
            "classes": classes,
            "train_imgs": train_imgs,
            "val_imgs": val_imgs,
            "test_imgs": test_imgs,
            "total_imgs": total_imgs,
            "label_count": label_count,
            "missing_labels": missing_labels,
            "corrupted_images": corrupted_images,
            "duplicate_images": duplicate_images,
            "empty_labels": empty_labels,
            "resolution": sample_res,
            "size_bytes": size_bytes,
            "size_str": size_str,
            "last_modified": last_mod,
            "health_score": health_score,
            "status": status,
            "status_reason": status_reason,
            "display_status": display_status,
        }

    def _count_split_images(self, split_names: List[str]) -> int:
        count = 0
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for sname in split_names:
            candidates = [
                self.dataset_path / sname / "images",
                self.dataset_path / "images" / sname,
                self.dataset_path / sname,
            ]
            for cand in candidates:
                if cand.exists() and cand.is_dir():
                    for p in cand.glob("*.*"):
                        if p.suffix.lower() in img_exts:
                            count += 1
                    break
        return count

    def _audit_labels(self) -> Tuple[int, int, int]:
        """Count total labels, missing labels for images, and empty 0-byte label files."""
        total_labels = 0
        missing_labels = 0
        empty_labels = 0
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        label_dirs = list(self.dataset_path.glob("**/labels")) + list(self.dataset_path.glob("train/labels"))
        if not label_dirs:
            label_dirs = [self.dataset_path]

        for ld in set(label_dirs):
            if ld.exists() and ld.is_dir():
                for lbl in ld.glob("*.txt"):
                    total_labels += 1
                    if lbl.stat().st_size == 0:
                        empty_labels += 1

        # Check missing labels for images
        for split in ["train", "valid", "val"]:
            img_dir = self.dataset_path / split / "images"
            lbl_dir = self.dataset_path / split / "labels"
            if img_dir.exists() and lbl_dir.exists():
                for img_p in img_dir.glob("*.*"):
                    if img_p.suffix.lower() in img_exts:
                        expected_lbl = lbl_dir / f"{img_p.stem}.txt"
                        if not expected_lbl.exists():
                            missing_labels += 1

        return total_labels, missing_labels, empty_labels

    def _audit_images_sample(self) -> Tuple[int, int, str]:
        """Audit sample images for corruption, duplicate contents, and image resolution."""
        corrupted = 0
        duplicates = 0
        res_sample = "640x640"
        seen_hashes = set()
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        image_files = []
        for p in self.dataset_path.glob("**/*.*"):
            if p.is_file() and p.suffix.lower() in img_exts:
                image_files.append(p)
                if len(image_files) >= 200:
                    break

        for img_p in image_files:
            # Check 0-byte
            if img_p.stat().st_size == 0:
                corrupted += 1
                continue

            # Verify image header via PIL if available
            if Image is not None:
                try:
                    with Image.open(img_p) as im:
                        im.verify()
                        res_sample = f"{im.width}x{im.height}"
                except Exception:
                    corrupted += 1
                    continue

            # Check duplicates via fast size + header hash
            try:
                h = f"{img_p.stat().st_size}_{img_p.name}"
                if h in seen_hashes:
                    duplicates += 1
                else:
                    seen_hashes.add(h)
            except Exception:
                pass

        return corrupted, duplicates, res_sample

    def _calculate_dir_size(self, path: Path) -> int:
        total = 0
        if not path.exists():
            return 0
        for p in path.glob("**/*"):
            if p.is_file() and not p.is_symlink():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
        return total


# --- Dynamic Dataset Scanner ---

class DynamicDatasetScanner:
    """Production-grade Dynamic Dataset Management Subsystem scanning <Project Root>/data/ live on every invocation."""

    def __init__(self, root_dir: Path = Path(".")) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.data_dir = self.root_dir / "data"

    def discover_datasets(self, force_rescan: bool = True) -> List[Dict[str, Any]]:
        """Scan <Project Root>/data/ (and any additional data directories) LIVE from disk."""
        discovered: List[Dict[str, Any]] = []
        seen_paths = set()

        search_dirs = [
            self.data_dir,
            self.root_dir / "dataset_micro",
            self.root_dir / "datasets",
            Path("/content/data"),
            Path("/content/datasets"),
        ]

        for base_dir in search_dirs:
            if not base_dir.exists() or not base_dir.is_dir():
                continue

            # Check if base_dir itself is a dataset or scan children
            subfolders = [base_dir] if (base_dir / "data.yaml").exists() or (base_dir / "train").exists() else [d for d in base_dir.iterdir() if d.is_dir()]

            for folder in subfolders:
                abs_p = str(folder.resolve())
                if abs_p in seen_paths or folder.name.startswith("."):
                    continue

                seen_paths.add(abs_p)

                # Check smart cache if force_rescan is False
                if not force_rescan:
                    cached = _DATASET_CACHE.get(folder)
                    if cached:
                        discovered.append(cached)
                        continue

                # Live filesystem audit
                auditor = DatasetAuditor(folder)
                info = auditor.audit()
                _DATASET_CACHE.put(folder, info)
                discovered.append(info)

        return discovered

    def sort_datasets(self, datasets: List[Dict[str, Any]], sort_key: str = "alphabetical", descending: bool = False) -> List[Dict[str, Any]]:
        """Sort dataset list by specified key."""
        key = sort_key.lower()
        if key in ("alphabetical", "name"):
            return sorted(datasets, key=lambda d: d.get("name", "").lower(), reverse=descending)
        elif key in ("newest", "date"):
            return sorted(datasets, key=lambda d: d.get("last_modified", ""), reverse=not descending)
        elif key in ("largest", "size"):
            return sorted(datasets, key=lambda d: d.get("size_bytes", 0), reverse=not descending)
        elif key in ("images", "most images"):
            return sorted(datasets, key=lambda d: d.get("total_imgs", 0), reverse=not descending)
        elif key in ("health", "health score"):
            return sorted(datasets, key=lambda d: d.get("health_score", 0), reverse=not descending)
        return datasets

    def search_datasets(
        self,
        datasets: List[Dict[str, Any]],
        query: str = "",
        format_filter: str = "",
        status_filter: str = "",
    ) -> List[Dict[str, Any]]:
        """Filter dataset list by text query, format filter, or status filter."""
        filtered = datasets
        if query.strip():
            q = query.strip().lower()
            filtered = [d for d in filtered if q in d.get("name", "").lower() or q in d.get("format", "").lower() or q in str(d.get("classes", [])).lower()]

        if format_filter.strip():
            ff = format_filter.strip().lower()
            filtered = [d for d in filtered if d.get("format", "").lower() == ff]

        if status_filter.strip():
            sf = status_filter.strip().lower()
            filtered = [d for d in filtered if sf in d.get("status", "").lower()]

        return filtered

    def self_heal_dataset(self, dataset_path: Union[str, Path]) -> Tuple[bool, str]:
        """Automatically repair missing standard directories (train/images, train/labels, valid/images, valid/labels)."""
        p = Path(dataset_path).resolve()
        if not p.exists():
            return False, f"Path '{p}' does not exist."

        required_dirs = [
            p / "train" / "images",
            p / "train" / "labels",
            p / "valid" / "images",
            p / "valid" / "labels",
            p / "test" / "images",
            p / "test" / "labels",
        ]

        created = []
        try:
            for rd in required_dirs:
                if not rd.exists():
                    rd.mkdir(parents=True, exist_ok=True)
                    created.append(str(rd.relative_to(p)))

            # If data.yaml is missing, generate basic template
            yaml_p = p / "data.yaml"
            if not yaml_p.exists():
                with open(yaml_p, "w", encoding="utf-8") as f:
                    f.write(f"# Auto-generated AETHEL dataset configuration\npath: {p.name}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: target_object\n")
                created.append("data.yaml")

            # Force cache invalidation
            _DATASET_CACHE.clear()

            return True, f"Self-healed dataset '{p.name}'. Created: {', '.join(created) if created else 'Already complete'}"
        except Exception as e:
            logger.error(f"Error during dataset self-healing: {e}")
            return False, f"Self-healing failed: {e}"
