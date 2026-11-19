"""Comprehensive Unit Test Suite for Aurelix Dataset Subsystem.

Tests parsers, loaders, validators, statistics, transforms, batch collators,
caching, and visualization managers.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aurelix.datasets.cache import DatasetCache
from aurelix.datasets.collator import BatchCollator
from aurelix.datasets.detection import DetectionDataset
from aurelix.datasets.loader import AnnotationLoader, ImageLoader
from aurelix.datasets.metadata import AnnotationInfo, ImageInfo
from aurelix.datasets.parser import DatasetParser, LabelParser
from aurelix.datasets.statistics import DatasetStatistics
from aurelix.datasets.transforms import TransformPipeline
from aurelix.datasets.validator import DatasetIntegrityChecker, DatasetValidator
from aurelix.datasets.visualization import VisualizationManager
from aurelix.registry.registry import DATASET_REGISTRY
from aurelix.utils.exceptions import DatasetException


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Create a mock YOLO format dataset directory structure."""
    dataset_dir = tmp_path / "mock_dataset"
    img_dir = dataset_dir / "train" / "images"
    lbl_dir = dataset_dir / "train" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create dummy images
    for i in range(3):
        img = Image.new("RGB", (640, 480), color=(100 + i * 20, 50, 50))
        img_path = img_dir / f"img_{i}.jpg"
        img.save(img_path)

        lbl_path = lbl_dir / f"img_{i}.txt"
        lbl_path.write_text(f"{i % 2} 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    return dataset_dir


def test_image_loader(tmp_path):
    """Verify ImageLoader loads images into uint8 NumPy arrays."""
    img_file = tmp_path / "test.jpg"
    Image.new("RGB", (100, 200), color=(255, 0, 0)).save(img_file)

    arr = ImageLoader.load_image(img_file)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (200, 100, 3)

    width, height = ImageLoader.get_image_size(img_file)
    assert width == 100
    assert height == 200


def test_label_parser(tmp_path):
    """Verify YOLO label file parsing."""
    lbl_file = tmp_path / "test.txt"
    lbl_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.2 0.4 0.1 0.1\n", encoding="utf-8")

    anns = LabelParser.parse_yolo_file(lbl_file)
    assert len(anns) == 2
    assert anns[0].class_id == 0
    assert anns[0].bbox == (0.5, 0.5, 0.2, 0.3)


def test_dataset_validator():
    """Verify DatasetValidator flags invalid class IDs and out-of-bounds coords."""
    validator = DatasetValidator(num_classes=10)

    invalid_ann = AnnotationInfo(class_id=15, bbox=(1.5, 0.5, 0.2, 0.3), format="xywh_norm")
    issues = validator.validate_annotation(invalid_ann)

    assert len(issues) >= 2
    assert any("Invalid class_id" in issue for issue in issues)
    assert any("out of [0, 1] bounds" in issue for issue in issues)


def test_dataset_statistics():
    """Verify DatasetStatistics generates JSON-serializable output."""
    sample = ImageInfo(
        image_path=Path("img.jpg"),
        height=480,
        width=640,
        annotations=[
            AnnotationInfo(class_id=0, bbox=(0.5, 0.5, 0.2, 0.3), format="xywh_norm"),
            AnnotationInfo(class_id=1, bbox=(0.2, 0.2, 0.1, 0.1), format="xywh_norm"),
        ],
    )
    stats = DatasetStatistics.compute([sample], class_names=["cat", "dog"])

    assert stats["total_images"] == 1
    assert stats["total_annotations"] == 2
    assert stats["completeness_score"] == 1.0
    assert "cat" in stats["class_distribution"]

    # Verify JSON serializability
    json_str = json.dumps(stats)
    assert "total_images" in json_str


def test_transform_pipeline():
    """Verify TransformPipeline resizes images to target imgsz."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    anns = [AnnotationInfo(class_id=0, bbox=(0.5, 0.5, 0.2, 0.3), format="xywh_norm")]

    pipeline = TransformPipeline(config={"imgsz": 640}, is_train=False)
    transformed_img, transformed_anns = pipeline(img, anns)

    assert transformed_img.shape == (640, 640, 3)
    assert len(transformed_anns) == 1


def test_dataset_cache():
    """Verify DatasetCache stores and retrieves NumPy arrays."""
    cache = DatasetCache(max_mb=10.0, enabled=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    cache.put("key1", img)
    retrieved = cache.get("key1")

    assert retrieved is not None
    assert retrieved.shape == (100, 100, 3)


def test_detection_dataset_integration(mock_dataset_dir):
    """Verify DetectionDataset loads samples from directory."""
    ds = DetectionDataset(
        root=mock_dataset_dir,
        format_type="yolo",
        split="train",
        config={"imgsz": 640},
    )

    assert len(ds) == 3
    img, anns = ds[0]
    assert img.shape == (640, 640, 3)
    assert len(anns) == 1

    # Verify registry lookup
    assert DATASET_REGISTRY.contains("DetectionDataset")


def test_batch_collator():
    """Verify BatchCollator processes batch lists."""
    img1 = np.zeros((640, 640, 3), dtype=np.uint8)
    img2 = np.zeros((640, 640, 3), dtype=np.uint8)
    ann1 = [AnnotationInfo(class_id=0, bbox=(0.1, 0.1, 0.5, 0.5), format="xyxy")]
    ann2 = [AnnotationInfo(class_id=1, bbox=(0.2, 0.2, 0.6, 0.6), format="xyxy")]

    batch = [(img1, ann1), (img2, ann2)]
    collator = BatchCollator()
    imgs_out, targets_out = collator(batch)

    assert imgs_out is not None
    assert targets_out is not None


def test_visualization_manager(tmp_path):
    """Verify VisualizationManager saves annotated preview images."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    anns = [AnnotationInfo(class_id=0, bbox=(0.5, 0.5, 0.2, 0.3), format="xywh_norm")]

    out_file = tmp_path / "preview.jpg"
    VisualizationManager.save_sample_preview(img, anns, out_file, class_names=["cat"])

    assert out_file.exists()
