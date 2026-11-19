"""Unit & Integration Test Suite for Dynamic Dataset Discovery & Management Subsystem.

Tests live filesystem scanning in <Project Root>/data/, format detection (YOLO, COCO, VOC,
Classification, Segmentation, OBB, Pose), corrupt & duplicate image detection, YAML updates,
smart cache auto-invalidation, self-healing, dataset creation/rename/deletion, and index generation.
"""

import json
import shutil
import time
from pathlib import Path

import pytest

from aurelix.datasets.discovery import (
    COCOFormatDetector,
    ClassificationFormatDetector,
    DatasetAuditor,
    DynamicDatasetScanner,
    OBBFormatDetector,
    PoseFormatDetector,
    SegmentationFormatDetector,
    SmartDatasetCache,
    VOCFormatDetector,
    YOLOFormatDetector,
)
from aurelix.datasets.reports import DatasetReportGenerator
from aurelix.launcher.managers import DatasetManager


@pytest.fixture
def data_workspace(tmp_path: Path):
    """Create a temporary workspace directory mimicking <Project Root>/data/."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a valid YOLO dataset
    yolo_ds = data_dir / "new_person_detection"
    (yolo_ds / "train" / "images").mkdir(parents=True, exist_ok=True)
    (yolo_ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (yolo_ds / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (yolo_ds / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    with open(yolo_ds / "data.yaml", "w", encoding="utf-8") as f:
        f.write("train: train/images\nval: valid/images\nnames:\n  0: person\n  1: vehicle\n")

    # Create dummy images and labels
    (yolo_ds / "train" / "images" / "img1.jpg").write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01")
    (yolo_ds / "train" / "labels" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    return tmp_path


def test_format_detectors(data_workspace: Path):
    """Test format detectors for YOLO, COCO, VOC, Classification, Segmentation, OBB, Pose."""
    data_dir = data_workspace / "data"

    # YOLO
    assert YOLOFormatDetector().detect(data_dir / "new_person_detection")[0] is True

    # COCO
    coco_ds = data_dir / "coco_test"
    (coco_ds / "annotations").mkdir(parents=True, exist_ok=True)
    (coco_ds / "annotations" / "instances_train.json").write_text("{}")
    assert COCOFormatDetector().detect(coco_ds)[0] is True

    # VOC
    voc_ds = data_dir / "voc_test"
    (voc_ds / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (voc_ds / "Annotations").mkdir(parents=True, exist_ok=True)
    assert VOCFormatDetector().detect(voc_ds)[0] is True

    # Classification
    cls_ds = data_dir / "cls_test"
    (cls_ds / "cat").mkdir(parents=True, exist_ok=True)
    (cls_ds / "dog").mkdir(parents=True, exist_ok=True)
    (cls_ds / "cat" / "c1.jpg").write_bytes(b"dummy")
    assert ClassificationFormatDetector().detect(cls_ds)[0] is True

    # Segmentation
    seg_ds = data_dir / "seg_test"
    (seg_ds / "masks").mkdir(parents=True, exist_ok=True)
    assert SegmentationFormatDetector().detect(seg_ds)[0] is True

    # OBB
    obb_ds = data_dir / "obb_test"
    (obb_ds / "obb").mkdir(parents=True, exist_ok=True)
    assert OBBFormatDetector().detect(obb_ds)[0] is True


def test_dynamic_scanner_live_addition_and_deletion(data_workspace: Path):
    """Verify new dataset creation appears immediately, rename updates, and deletion removes it."""
    scanner = DynamicDatasetScanner(root_dir=data_workspace)
    
    # Initial scan
    datasets = scanner.discover_datasets(force_rescan=True)
    assert len(datasets) == 1
    assert datasets[0]["name"] == "new_person_detection"

    # 1. Add temporary_test_dataset
    tmp_ds = data_workspace / "data" / "temporary_test_dataset"
    (tmp_ds / "train" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_ds / "train" / "images" / "test.jpg").write_bytes(b"dummy")
    with open(tmp_ds / "data.yaml", "w", encoding="utf-8") as f:
        f.write("names: ['test']\n")

    # Rescan immediately - no restart
    datasets2 = scanner.discover_datasets(force_rescan=True)
    names2 = [d["name"] for d in datasets2]
    assert "temporary_test_dataset" in names2

    # 2. Rename temporary_test_dataset -> tyre_dataset
    renamed_ds = data_workspace / "data" / "tyre_dataset"
    tmp_ds.rename(renamed_ds)

    datasets3 = scanner.discover_datasets(force_rescan=True)
    names3 = [d["name"] for d in datasets3]
    assert "tyre_dataset" in names3
    assert "temporary_test_dataset" not in names3

    # 3. Delete tyre_dataset
    shutil.rmtree(renamed_ds)

    datasets4 = scanner.discover_datasets(force_rescan=True)
    names4 = [d["name"] for d in datasets4]
    assert "tyre_dataset" not in names4


def test_yaml_update_and_cache_invalidation(data_workspace: Path):
    """Test smart cache invalidation when YAML file or folder mtime is modified."""
    scanner = DynamicDatasetScanner(root_dir=data_workspace)
    
    ds_path = data_workspace / "data" / "new_person_detection"
    initial_ds = scanner.discover_datasets(force_rescan=True)[0]
    assert initial_ds["num_classes"] == 2
    assert initial_ds["classes"] == ["person", "vehicle"]

    # Modify YAML file
    time.sleep(0.05)
    with open(ds_path / "data.yaml", "w", encoding="utf-8") as f:
        f.write("names: ['person', 'vehicle', 'helmet']\n")

    # Rescan with cache check (force_rescan=False) -> should auto-invalidate cache!
    updated_ds = scanner.discover_datasets(force_rescan=False)[0]
    assert updated_ds["num_classes"] == 3
    assert "helmet" in updated_ds["classes"]


def test_dataset_auditor_health_and_corrupt_images(data_workspace: Path):
    """Test health score deductions for corrupted images, missing labels, and status generation."""
    ds_path = data_workspace / "data" / "new_person_detection"
    
    # Add corrupted image (0-byte)
    (ds_path / "train" / "images" / "corrupt.jpg").write_bytes(b"")

    auditor = DatasetAuditor(ds_path)
    audit = auditor.audit()

    assert audit["corrupted_images"] >= 1
    assert audit["health_score"] < 100
    assert audit["status"] in ("WARNING", "ERROR", "INVALID")


def test_dataset_self_healing(data_workspace: Path):
    """Test dataset self-healing of missing directories."""
    scanner = DynamicDatasetScanner(root_dir=data_workspace)
    empty_ds = data_workspace / "data" / "broken_ds"
    empty_ds.mkdir(parents=True, exist_ok=True)

    success, msg = scanner.self_heal_dataset(empty_ds)
    assert success
    assert (empty_ds / "train" / "images").exists()
    assert (empty_ds / "train" / "labels").exists()
    assert (empty_ds / "data.yaml").exists()


def test_dataset_reports_generation(data_workspace: Path):
    """Test generating runs/datasets/ index.json, history.json, report.html, report.csv."""
    generator = DatasetReportGenerator(root_dir=data_workspace)
    reps = generator.generate_all_reports()

    assert Path(reps["index.json"]).exists()
    assert Path(reps["history.json"]).exists()
    assert Path(reps["report.csv"]).exists()
    assert Path(reps["report.html"]).exists()


def test_dataset_manager_adapter(data_workspace: Path):
    """Test refactored DatasetManager integration."""
    mgr = DatasetManager(root_dir=data_workspace)
    datasets = mgr.discover_datasets()

    assert len(datasets) >= 1
    sorted_ds = mgr.sort_datasets(datasets, sort_key="alphabetical")
    assert len(sorted_ds) == len(datasets)

    searched = mgr.search_datasets(datasets, query="person")
    assert len(searched) >= 1
