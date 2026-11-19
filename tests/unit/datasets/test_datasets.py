"""Unit tests for datasets subpackage (YOLODataset, COCODataset, VOCDataset, CustomDataset)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
from torch.utils.data import DataLoader

from datasets import COCODataset, CustomDataset, VOCDataset, YOLODataset


def test_yolo_dataset(tmp_path: any) -> None:
    """Test YOLODataset text annotation loader."""
    img_path = tmp_path / "img1.jpg"
    img_path.write_text("dummy")

    lbl_path = tmp_path / "img1.txt"
    lbl_path.write_text("0 0.5 0.5 0.4 0.4\n2 0.2 0.2 0.1 0.1\n")

    ds = YOLODataset(img_paths=[img_path], label_paths=[lbl_path], img_size=(320, 320), max_gt=10)
    assert len(ds) == 1

    img, labels, boxes, mask = ds[0]
    assert img.shape == torch.Size([3, 320, 320])
    assert labels.shape == torch.Size([10, 1])
    assert boxes.shape == torch.Size([10, 4])
    assert mask.shape == torch.Size([10, 1])
    assert mask[:2].sum().item() == 2.0


def test_coco_dataset(tmp_path: any) -> None:
    """Test COCODataset JSON annotation loader."""
    json_path = tmp_path / "coco.json"
    coco_data = {
        "images": [{"id": 1, "filename": "img1.jpg", "width": 640, "height": 480}],
        "annotations": [{"image_id": 1, "category_id": 5, "bbox": [10, 10, 100, 100]}],
    }
    json_path.write_text(json.dumps(coco_data))

    ds = COCODataset(annotation_json=json_path, img_dir=tmp_path, img_size=(320, 320), max_gt=10)
    assert len(ds) == 1

    img, labels, boxes, mask = ds[0]
    assert mask[0].item() == 1.0
    assert labels[0].item() == 5.0


def test_voc_dataset(tmp_path: any) -> None:
    """Test VOCDataset XML annotation loader."""
    xml_path = tmp_path / "img1.xml"
    xml_content = """<annotation>
    <size><width>640</width><height>480</height></size>
    <object>
        <name>cat</name>
        <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>100</xmax><ymax>100</ymax></bndbox>
    </object>
</annotation>"""
    xml_path.write_text(xml_content)

    ds = VOCDataset(xml_paths=[xml_path], img_size=(320, 320), class_to_idx={"cat": 1}, max_gt=10)
    assert len(ds) == 1

    img, labels, boxes, mask = ds[0]
    assert mask[0].item() == 1.0
    assert labels[0].item() == 1.0


def test_custom_dataset_dataloader() -> None:
    """Test CustomDataset loader and DataLoader collation."""
    samples = [
        {
            "image": torch.randn(3, 320, 320),
            "boxes": [[10.0, 10.0, 50.0, 50.0]],
            "labels": [2.0],
        },
        {
            "image": torch.randn(3, 320, 320),
            "boxes": [[20.0, 20.0, 80.0, 80.0]],
            "labels": [0.0],
        },
    ]

    ds = CustomDataset(samples=samples, img_size=(320, 320), max_gt=10)
    loader = DataLoader(ds, batch_size=2)

    for images, labels, boxes, masks in loader:
        assert images.shape == torch.Size([2, 3, 320, 320])
        assert labels.shape == torch.Size([2, 10, 1])
        assert boxes.shape == torch.Size([2, 10, 4])
        assert masks.shape == torch.Size([2, 10, 1])
