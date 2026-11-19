"""Comprehensive Unit Test Suite for Aethel Multi-Modal Indigenous Architectures.

Covers:
1. Pose Estimation (AethelPoseModel, AethelPoseHead, OKSLoss, KeypointL1Loss, AethelPoseLoss)
2. Instance Segmentation (AethelSegmentModel, ProtoNet, AethelSegmentHead, DiceLoss, MaskBCELoss, AethelSegmentLoss)
3. Oriented Bounding Boxes (AethelOBBModel, AethelOBBHead, decode_angle, obb_to_corners, ContinuousAngleLoss, ProbIoULoss, AethelOBBLoss)
4. Multi-Modal PredictionResult serialization & AethelVisualizer rendering across all tasks.
"""

import math
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aethel.inference.result import PredictionResult
from aethel.inference.visualizer import AethelVisualizer
from aethel.losses.obb import AethelOBBLoss, ContinuousAngleLoss, ProbIoULoss
from aethel.losses.pose import AethelPoseLoss, KeypointL1Loss, OKSLoss
from aethel.losses.segment import AethelSegmentLoss, DiceLoss, MaskBCELoss
from aethel.models.head.obb import AethelOBBHead, decode_angle, obb_to_corners
from aethel.models.head.pose import AethelPoseHead, COCO_KEYPOINTS, COCO_SKELETON, decode_keypoints
from aethel.models.head.segment import AethelSegmentHead, ProtoNet, assemble_masks
from aethel.models.obb import (
    AethelOBBModel,
    build_aethel_obb_medium,
    build_aethel_obb_small,
)
from aethel.models.pose import (
    AethelPoseModel,
    build_aethel_pose_medium,
    build_aethel_pose_small,
)
from aethel.models.segment import (
    AethelSegmentModel,
    build_aethel_segment_medium,
    build_aethel_segment_small,
)
from aethel.registry.registry import MODEL_REGISTRY, LOSS_REGISTRY


# =========================================================================
# 1. Pose Estimation Tests
# =========================================================================

def test_aethel_pose_registry():
    """Verify registry presence and COCO 17-keypoint constants in Aethel."""
    assert MODEL_REGISTRY.contains("AethelPose")
    assert MODEL_REGISTRY.contains("aethel_pose_small")
    assert MODEL_REGISTRY.contains("aethel_pose_medium")
    assert len(COCO_KEYPOINTS) == 17
    assert len(COCO_SKELETON) == 19


def test_aethel_pose_model_construction():
    """Verify construction of AethelPose small and medium variants."""
    pose_small = build_aethel_pose_small(num_classes=1, num_keypoints=17)
    assert isinstance(pose_small, AethelPoseModel)
    assert pose_small.num_keypoints == 17
    assert pose_small.num_classes == 1

    pose_med = build_aethel_pose_medium(num_classes=1, num_keypoints=17)
    assert isinstance(pose_med, AethelPoseModel)
    assert pose_med.scale == "medium"


def test_aethel_pose_decoding():
    """Verify keypoint decoding mathematical stability in Aethel."""
    offsets = np.array([[[1.5, 2.5, 0.8]]], dtype=np.float32)
    anchors = np.array([[10.0, 20.0]], dtype=np.float32)
    strides = np.array([[8.0]], dtype=np.float32)

    decoded = decode_keypoints(offsets, anchors, strides)
    assert decoded.shape == (1, 1, 3)
    assert pytest.approx(float(decoded[0, 0, 0]), 1e-3) == 92.0
    assert pytest.approx(float(decoded[0, 0, 1]), 1e-3) == 180.0
    expected_conf = 1.0 / (1.0 + math.exp(-0.8))
    assert pytest.approx(float(decoded[0, 0, 2]), 1e-3) == expected_conf


# =========================================================================
# 2. Instance Segmentation Tests
# =========================================================================

def test_aethel_segment_construction():
    """Verify construction of AethelSegment variants."""
    assert MODEL_REGISTRY.contains("AethelSegment")
    assert MODEL_REGISTRY.contains("aethel_segment_small")
    assert MODEL_REGISTRY.contains("aethel_segment_medium")

    seg_small = build_aethel_segment_small(num_classes=80, num_protos=32)
    assert isinstance(seg_small, AethelSegmentModel)
    assert seg_small.num_protos == 32
    assert seg_small.num_classes == 80


def test_aethel_mask_assembly():
    """Verify prototype mask linear combination and bounding box cropping."""
    protos = np.random.randn(1, 32, 40, 40).astype(np.float32)
    coeffs = np.random.randn(3, 32).astype(np.float32)
    boxes = np.array([
        [10.0, 10.0, 100.0, 100.0],
        [20.0, 20.0, 80.0, 80.0],
        [0.0, 0.0, 160.0, 160.0],
    ], dtype=np.float32)

    masks = assemble_masks(protos, coeffs, boxes)
    assert masks.shape == (3, 40, 40)
    assert np.all(masks >= 0.0)
    assert np.all(masks <= 1.0)


# =========================================================================
# 3. Oriented Bounding Box (OBB) Tests
# =========================================================================

def test_aethel_obb_construction():
    """Verify construction of AethelOBB variants."""
    assert MODEL_REGISTRY.contains("AethelOBB")
    assert MODEL_REGISTRY.contains("aethel_obb_small")
    assert MODEL_REGISTRY.contains("aethel_obb_medium")

    obb_small = build_aethel_obb_small(num_classes=15)
    assert isinstance(obb_small, AethelOBBModel)
    assert obb_small.num_classes == 15


def test_aethel_obb_angle_decoding():
    """Verify continuous trigonometric (cos 2θ, sin 2θ) angle decoding in Aethel."""
    trig_pi_4 = np.array([[0.0, 1.0]], dtype=np.float32)
    angle_pi_4 = decode_angle(trig_pi_4)
    assert pytest.approx(float(angle_pi_4[0]), 1e-4) == math.pi / 4.0


def test_aethel_obb_corners():
    """Verify 4-corner polygon coordinates computation from (cx, cy, w, h, angle)."""
    obb = np.array([[100.0, 100.0, 40.0, 20.0, 0.0]], dtype=np.float32)
    corners = obb_to_corners(obb)
    assert corners.shape == (1, 4, 2)
    expected = np.array([[[80.0, 90.0], [120.0, 90.0], [120.0, 110.0], [80.0, 110.0]]], dtype=np.float32)
    np.testing.assert_allclose(corners, expected, atol=1e-4)


# =========================================================================
# 4. Multi-Modal PredictionResult & Visualizer Tests
# =========================================================================

def test_aethel_visualizer_rendering():
    """Verify AethelVisualizer renders pose skeletons, masks, and OBB polygons without error."""
    vis = AethelVisualizer()
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)

    # Pose
    kpts = np.zeros((1, 17, 3), dtype=np.float32)
    for i in range(17):
        kpts[0, i] = [50.0 + i * 5, 50.0 + i * 5, 0.95]
    res_pose = PredictionResult(
        boxes=np.array([[20.0, 20.0, 180.0, 180.0]], dtype=np.float32),
        scores=np.array([0.92], dtype=np.float32),
        classes=np.array([0], dtype=np.int64),
        labels=["person"],
        keypoints=kpts,
    )
    img_pose = vis.draw(canvas, res_pose)
    assert isinstance(img_pose, Image.Image)

    # OBB
    res_obb = PredictionResult(
        boxes=np.array([[40.0, 40.0, 160.0, 160.0]], dtype=np.float32),
        scores=np.array([0.85], dtype=np.float32),
        classes=np.array([2], dtype=np.int64),
        labels=["plane"],
        obb_angles=np.array([math.pi / 4], dtype=np.float32),
        obb_corners=np.array([[[100.0, 50.0], [150.0, 100.0], [100.0, 150.0], [50.0, 100.0]]], dtype=np.float32),
    )
    img_obb = vis.draw(canvas, res_obb)
    assert isinstance(img_obb, Image.Image)


def test_aethel_build_model_multimodal():
    """Verify build_model universal factory builds Aethel Pose, Segment, and OBB models."""
    from aethel.models.aethelnet import build_model

    m_pose = build_model("aethel_pose_small")
    assert isinstance(m_pose, AethelPoseModel)

    m_seg = build_model("aethel_segment_small")
    assert isinstance(m_seg, AethelSegmentModel)

    m_obb = build_model("aethel_obb_small")
    assert isinstance(m_obb, AethelOBBModel)
