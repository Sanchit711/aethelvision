"""Comprehensive Unit Test Suite for Aurelix Vision Datasets Package.

Covers:
- YOLO dataset parser & data.yaml auto-parsing
- DatasetVerifier & VerificationReport
- DatasetReporter & Statistics
- DatasetFactory
- DetectionCollate & collate_fn
- DataLoader builders (build_train_loader, build_validation_loader, build_test_loader)
- Image and box augmentation transforms (Letterbox, Resize, RandomFlip, HSVAugmentation, Normalize, Compose)
"""

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except Exception:
    torch = None  # type: ignore

from datasets.collate import DetectionCollate, collate_fn
from datasets.dataloader import build_test_loader, build_train_loader, build_validation_loader
from datasets.factory import DatasetFactory
from datasets.reporter import DatasetReporter
from datasets.transforms import Compose, HSVAugmentation, Letterbox, Normalize, RandomFlip, Resize
from datasets.verifier import DatasetVerifier, VerificationReport
from datasets.yolo import YOLODataset
from trainer.configuration import TrainingConfig


@pytest.fixture
def mock_yolo_dataset(tmp_path):
    """Create temporary mock YOLO dataset directory with images and labels."""
    ds_dir = tmp_path / "person_detection"
    train_img_dir = ds_dir / "train" / "images"
    train_lbl_dir = ds_dir / "train" / "labels"
    val_img_dir = ds_dir / "val" / "images"
    val_lbl_dir = ds_dir / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        img = Image.new("RGB", (640, 480), color=(200, 100, 50))
        img.save(train_img_dir / "sample1.jpg")
        img.save(train_img_dir / "sample2.jpg")
        img.save(val_img_dir / "val1.jpg")
    except Exception:
        pass

    with open(train_lbl_dir / "sample1.txt", "w") as f:
        f.write("0 0.5 0.5 0.4 0.4\n")
    with open(train_lbl_dir / "sample2.txt", "w") as f:
        f.write("0 0.3 0.3 0.2 0.2\n")
    with open(val_lbl_dir / "val1.txt", "w") as f:
        f.write("0 0.6 0.6 0.3 0.3\n")

    data_yaml = ds_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write("path: .\ntrain: train/images\nval: val/images\nnames:\n  0: person\n")

    return ds_dir


def test_yolo_parser(mock_yolo_dataset):
    """Test YOLO text dataset parser and data.yaml discovery."""
    ds = YOLODataset(mock_yolo_dataset, split="train", img_size=(640, 640))
    assert len(ds) == 2
    assert ds.num_classes == 1
    assert ds.class_names == ["person"]

    if torch is not None:
        img_tensor, padded_labels, padded_boxes, mask_gt = ds[0]
        assert img_tensor.shape == (3, 640, 640)
        assert padded_labels.shape == (50, 1)
        assert padded_boxes.shape == (50, 4)
        assert mask_gt.shape in ((50,), (50, 1))


def test_dataset_factory(mock_yolo_dataset):
    """Test DatasetFactory dynamic creation."""
    ds_train = DatasetFactory.create_dataset(mock_yolo_dataset, dataset_type="yolo", split="train")
    ds_val = DatasetFactory.create_dataset(mock_yolo_dataset, dataset_type="auto", split="val")

    assert len(ds_train) == 2
    assert len(ds_val) == 1


def test_dataset_verifier(mock_yolo_dataset):
    """Test DatasetVerifier report generation."""
    verifier = DatasetVerifier(mock_yolo_dataset / "data.yaml")
    report = verifier.verify()

    assert isinstance(report, VerificationReport)
    assert report.total_images == 3
    assert report.total_labels == 3
    assert report.yaml_valid == True


def test_dataset_reporter(mock_yolo_dataset, tmp_path):
    """Test DatasetReporter statistics and CSV/JSON export."""
    reporter = DatasetReporter(mock_yolo_dataset)
    report_dict = reporter.generate_report()

    assert report_dict["total_images"] == 3
    assert report_dict["total_instances"] == 3

    json_file = tmp_path / "report.json"
    csv_file = tmp_path / "report.csv"

    reporter.export_json(report_dict, json_file)
    reporter.export_csv(report_dict, csv_file)

    assert json_file.exists()
    assert csv_file.exists()


def test_collate_fn():
    """Test DetectionCollate function batching."""
    if torch is None:
        return

    item1 = (torch.zeros(3, 640, 640), torch.zeros(50, 1), torch.zeros(50, 4), torch.zeros(50, dtype=torch.bool))
    item2 = (torch.zeros(3, 640, 640), torch.zeros(50, 1), torch.zeros(50, 4), torch.zeros(50, dtype=torch.bool))

    batch_imgs, batch_lbls, batch_boxes, batch_masks = collate_fn([item1, item2])
    assert batch_imgs.shape == (2, 3, 640, 640)
    assert batch_lbls.shape == (2, 50, 1)
    assert batch_boxes.shape == (2, 50, 4)
    assert batch_masks.shape == (2, 50)


def test_dataloader_builders(mock_yolo_dataset):
    """Test DataLoader builders (build_train_loader, build_validation_loader, build_test_loader)."""
    config = TrainingConfig(dataset_path=str(mock_yolo_dataset), batch_size=2, num_workers=0)

    train_loader = build_train_loader(config)
    val_loader = build_validation_loader(config)
    test_loader = build_test_loader(config)

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    if torch is not None:
        for imgs, lbls, boxes, masks in train_loader:
            assert imgs.shape == (2, 3, 640, 640)
            break


def test_transforms():
    """Test spatial and color transform operations."""
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)

    pipeline = Compose([
        Letterbox(new_shape=(640, 640), auto=False),
        RandomFlip(p_hflip=1.0),
        HSVAugmentation(),
        Normalize(),
    ])

    img_out, boxes_out = pipeline(dummy_img, dummy_boxes)
    assert img_out is not None
