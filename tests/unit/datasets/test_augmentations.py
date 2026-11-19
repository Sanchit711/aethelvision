"""Unit tests for object detection data augmentations in datasets/augmentations.py."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch

from datasets.augmentations import (
    HSVColorJitter,
    Letterbox,
    MixUp,
    Mosaic,
    RandomAffine,
    RandomFlip,
    Resize,
)


def test_letterbox() -> None:
    """Test Letterbox aspect-ratio preserving resize and padding."""
    letterbox = Letterbox(target_size=(640, 640))
    img = torch.randn(3, 480, 640)
    boxes = torch.tensor([[10.0, 10.0, 100.0, 100.0]])

    padded_img, scaled_boxes = letterbox(img, boxes)

    assert padded_img.shape == torch.Size([3, 640, 640])
    assert scaled_boxes.shape == boxes.shape
    assert scaled_boxes[0, 1] > 10.0  # Vertically padded shift applied


def test_resize() -> None:
    """Test direct spatial Resize module."""
    resize = Resize(target_size=(320, 320))
    img = torch.randn(3, 640, 640)
    boxes = torch.tensor([[20.0, 20.0, 100.0, 100.0]])

    resized_img, scaled_boxes = resize(img, boxes)

    assert resized_img.shape == torch.Size([3, 320, 320])
    assert torch.equal(scaled_boxes[0], torch.tensor([10.0, 10.0, 50.0, 50.0]))


def test_random_flip() -> None:
    """Test RandomFlip horizontal box coordinate inversion."""
    flipper = RandomFlip(p_h=1.0, p_v=0.0)
    img = torch.randn(3, 320, 320)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])  # w=320 -> x1'=270, x2'=310

    flipped_img, flipped_boxes = flipper(img, boxes)

    assert flipped_img.shape == img.shape
    assert flipped_boxes[0, 0].item() == 270.0
    assert flipped_boxes[0, 2].item() == 310.0


def test_hsv_color_jitter() -> None:
    """Test HSVColorJitter module."""
    jitter = HSVColorJitter()
    img = torch.rand(3, 320, 320)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])

    jittered_img, output_boxes = jitter(img, boxes)

    assert jittered_img.shape == img.shape
    assert torch.equal(output_boxes, boxes)


def test_random_affine() -> None:
    """Test RandomAffine scaling and translation transformation."""
    affine = RandomAffine(translate=0.1, scale=0.1)
    img = torch.randn(3, 320, 320)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])

    out_img, out_boxes = affine(img, boxes)

    assert out_img.shape == img.shape
    assert out_boxes.shape == boxes.shape


def test_mosaic() -> None:
    """Test 4-image Mosaic blending augmentation."""
    mosaic = Mosaic(target_size=(640, 640))
    imgs = [torch.randn(3, 320, 320) for _ in range(4)]
    boxes_list = [torch.tensor([[10.0, 10.0, 50.0, 50.0]]) for _ in range(4)]
    labels_list = [torch.tensor([[1.0]]) for _ in range(4)]

    mosaic_img, combined_boxes, combined_labels = mosaic(imgs, boxes_list, labels_list)

    assert mosaic_img.shape == torch.Size([3, 640, 640])
    assert combined_boxes.shape == torch.Size([4, 4])
    assert combined_labels.shape == torch.Size([4, 1])


def test_mixup() -> None:
    """Test 2-image MixUp alpha blending augmentation."""
    mixup = MixUp(alpha=0.5)
    img1, img2 = torch.randn(3, 320, 320), torch.randn(3, 320, 320)
    boxes1, boxes2 = torch.tensor([[10.0, 10.0, 50.0, 50.0]]), torch.tensor([[20.0, 20.0, 80.0, 80.0]])
    labels1, labels2 = torch.tensor([[0.0]]), torch.tensor([[2.0]])

    blended_img, cat_boxes, cat_labels = mixup(img1, boxes1, labels1, img2, boxes2, labels2)

    assert blended_img.shape == torch.Size([3, 320, 320])
    assert cat_boxes.shape == torch.Size([2, 4])
    assert cat_labels.shape == torch.Size([2, 1])
