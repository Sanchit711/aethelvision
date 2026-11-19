"""Comprehensive Unit Test Suite for Aurelix Multi-Modal Indigenous Architectures.

Covers:
1. Pose Estimation (AurelixPoseModel, AurelixPoseHead, OKSLoss, KeypointL1Loss, AurelixPoseLoss)
2. Instance Segmentation (AurelixSegmentModel, ProtoNet, AurelixSegmentHead, DiceLoss, MaskBCELoss, AurelixSegmentLoss)
3. Oriented Bounding Boxes (AurelixOBBModel, AurelixOBBHead, decode_angle, obb_to_corners, ContinuousAngleLoss, ProbIoULoss, AurelixOBBLoss)
4. Multi-Modal PredictionResult serialization & AurelixVisualizer rendering across all tasks.
"""

import math
import numpy as np
import pytest
from PIL import Image

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aurelix.inference.result import PredictionResult
from aurelix.inference.visualizer import AurelixVisualizer
from aurelix.losses.obb import AurelixOBBLoss, ContinuousAngleLoss, ProbIoULoss
from aurelix.losses.pose import AurelixPoseLoss, KeypointL1Loss, OKSLoss
from aurelix.losses.segment import AurelixSegmentLoss, DiceLoss, MaskBCELoss
from aurelix.models.head.obb import AurelixOBBHead, decode_angle, obb_to_corners
from aurelix.models.head.pose import AurelixPoseHead, COCO_KEYPOINTS, COCO_SKELETON, decode_keypoints
from aurelix.models.head.segment import AurelixSegmentHead, ProtoNet, assemble_masks
from aurelix.models.obb import (
    AurelixOBBModel,
    build_aurelix_obb_medium,
    build_aurelix_obb_small,
)
from aurelix.models.pose import (
    AurelixPoseModel,
    build_aurelix_pose_medium,
    build_aurelix_pose_small,
)
from aurelix.models.segment import (
    AurelixSegmentModel,
    build_aurelix_segment_medium,
    build_aurelix_segment_small,
)
from aurelix.registry.registry import MODEL_REGISTRY, LOSS_REGISTRY


# =========================================================================
# 1. Pose Estimation Tests
# =========================================================================

def test_pose_registry_and_constants():
    """Verify registry presence and COCO 17-keypoint constants."""
    assert MODEL_REGISTRY.contains("AurelixPose")
    assert MODEL_REGISTRY.contains("aurelix_pose_small")
    assert MODEL_REGISTRY.contains("aurelix_pose_medium")
    assert len(COCO_KEYPOINTS) == 17
    assert len(COCO_SKELETON) == 19
    assert ("left_shoulder", "left_elbow") in [
        (COCO_KEYPOINTS[j1], COCO_KEYPOINTS[j2]) for j1, j2 in COCO_SKELETON
    ]


def test_pose_model_construction():
    """Verify construction of AurelixPose small and medium variants."""
    pose_small = build_aurelix_pose_small(num_classes=1, num_keypoints=17)
    assert isinstance(pose_small, AurelixPoseModel)
    assert pose_small.num_keypoints == 17
    assert pose_small.num_classes == 1

    pose_med = build_aurelix_pose_medium(num_classes=1, num_keypoints=17)
    assert isinstance(pose_med, AurelixPoseModel)
    assert pose_med.scale == "medium"


def test_pose_head_and_decoding():
    """Verify keypoint decoding mathematical stability."""
    offsets = np.array([[[1.5, 2.5, 0.8]]], dtype=np.float32)
    anchors = np.array([[10.0, 20.0]], dtype=np.float32)
    strides = np.array([[8.0]], dtype=np.float32)

    decoded = decode_keypoints(offsets, anchors, strides)
    assert decoded.shape == (1, 1, 3)
    # x = (10 + 1.5) * 8 = 92, y = (20 + 2.5) * 8 = 180, conf = sigmoid(0.8) ~ 0.68997
    assert pytest.approx(float(decoded[0, 0, 0]), 1e-3) == 92.0
    assert pytest.approx(float(decoded[0, 0, 1]), 1e-3) == 180.0
    expected_conf = 1.0 / (1.0 + math.exp(-0.8))
    assert pytest.approx(float(decoded[0, 0, 2]), 1e-3) == expected_conf


def test_pose_losses():
    """Verify OKSLoss, KeypointL1Loss, and AurelixPoseLoss execution."""
    assert LOSS_REGISTRY.contains("OKSLoss")
    assert LOSS_REGISTRY.contains("KeypointL1Loss")
    assert LOSS_REGISTRY.contains("AurelixPoseLoss")

    oks_loss = OKSLoss()
    l1_loss = KeypointL1Loss()
    pose_loss = AurelixPoseLoss()

    # Fallback or torch computation
    if torch is not None:
        pred_kpts = torch.randn(2, 17, 2)
        target_kpts = torch.randn(2, 17, 2)
        areas = torch.tensor([[1000.0], [1500.0]])
        vis = torch.ones(2, 17)

        val_oks = oks_loss(pred_kpts, target_kpts, areas, vis)
        assert isinstance(val_oks, torch.Tensor)
        assert val_oks.item() >= 0.0

        val_l1 = l1_loss(pred_kpts, target_kpts, vis)
        assert isinstance(val_l1, torch.Tensor)
        assert val_l1.item() >= 0.0
    else:
        assert oks_loss(None, None, None) == 0.0
        total, bd = pose_loss({}, {})
        assert total == 0.0


# =========================================================================
# 2. Instance Segmentation Tests
# =========================================================================

def test_segment_registry_and_construction():
    """Verify registry presence and construction of AurelixSegment variants."""
    assert MODEL_REGISTRY.contains("AurelixSegment")
    assert MODEL_REGISTRY.contains("aurelix_segment_small")
    assert MODEL_REGISTRY.contains("aurelix_segment_medium")

    seg_small = build_aurelix_segment_small(num_classes=80, num_protos=32)
    assert isinstance(seg_small, AurelixSegmentModel)
    assert seg_small.num_protos == 32
    assert seg_small.num_classes == 80


def test_mask_assembly():
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
    # Mask values must be bounded between 0 and 1 via sigmoid
    assert np.all(masks >= 0.0)
    assert np.all(masks <= 1.0)


def test_segment_losses():
    """Verify DiceLoss, MaskBCELoss, and AurelixSegmentLoss."""
    assert LOSS_REGISTRY.contains("DiceLoss")
    assert LOSS_REGISTRY.contains("MaskBCELoss")
    assert LOSS_REGISTRY.contains("AurelixSegmentLoss")

    dice_loss = DiceLoss()
    bce_loss = MaskBCELoss()
    seg_loss = AurelixSegmentLoss()

    if torch is not None:
        pred_masks = torch.sigmoid(torch.randn(2, 40, 40))
        target_masks = (torch.rand(2, 40, 40) > 0.5).float()

        d_val = dice_loss(pred_masks, target_masks)
        b_val = bce_loss(pred_masks, target_masks)
        assert d_val.item() >= 0.0
        assert b_val.item() >= 0.0
    else:
        assert dice_loss(None, None) == 0.0
        total, bd = seg_loss({}, {})
        assert total == 0.0


# =========================================================================
# 3. Oriented Bounding Box (OBB) Tests
# =========================================================================

def test_obb_registry_and_construction():
    """Verify registry presence and construction of AurelixOBB variants."""
    assert MODEL_REGISTRY.contains("AurelixOBB")
    assert MODEL_REGISTRY.contains("aurelix_obb_small")
    assert MODEL_REGISTRY.contains("aurelix_obb_medium")

    obb_small = build_aurelix_obb_small(num_classes=15)
    assert isinstance(obb_small, AurelixOBBModel)
    assert obb_small.num_classes == 15


def test_obb_continuous_angle_decoding():
    """Verify continuous trigonometric (cos 2θ, sin 2θ) angle decoding."""
    # Test angle = 0 rad
    trig_0 = np.array([[1.0, 0.0]], dtype=np.float32)
    angle_0 = decode_angle(trig_0)
    assert pytest.approx(float(angle_0[0]), 1e-4) == 0.0

    # Test angle = pi/4 (45 deg -> 2*theta = pi/2 -> cos=0, sin=1)
    trig_pi_4 = np.array([[0.0, 1.0]], dtype=np.float32)
    angle_pi_4 = decode_angle(trig_pi_4)
    assert pytest.approx(float(angle_pi_4[0]), 1e-4) == math.pi / 4.0

    # Test angle = -pi/4 (-45 deg -> 2*theta = -pi/2 -> cos=0, sin=-1)
    trig_neg_pi_4 = np.array([[0.0, -1.0]], dtype=np.float32)
    angle_neg_pi_4 = decode_angle(trig_neg_pi_4)
    assert pytest.approx(float(angle_neg_pi_4[0]), 1e-4) == -math.pi / 4.0


def test_obb_to_corners_conversion():
    """Verify 4-corner polygon coordinates computation from (cx, cy, w, h, angle)."""
    # Centered at (100, 100), width=40, height=20, angle=0 rad
    obb = np.array([[100.0, 100.0, 40.0, 20.0, 0.0]], dtype=np.float32)
    corners = obb_to_corners(obb)

    assert corners.shape == (1, 4, 2)
    # With angle 0: corners should be (80, 90), (120, 90), (120, 110), (80, 110)
    expected = np.array([[[80.0, 90.0], [120.0, 90.0], [120.0, 110.0], [80.0, 110.0]]], dtype=np.float32)
    np.testing.assert_allclose(corners, expected, atol=1e-4)


def test_obb_losses():
    """Verify ContinuousAngleLoss, ProbIoULoss, and AurelixOBBLoss."""
    assert LOSS_REGISTRY.contains("ContinuousAngleLoss")
    assert LOSS_REGISTRY.contains("ProbIoULoss")
    assert LOSS_REGISTRY.contains("AurelixOBBLoss")

    ang_loss = ContinuousAngleLoss()
    prob_loss = ProbIoULoss()
    obb_loss = AurelixOBBLoss()

    if torch is not None:
        p_trig = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        t_ang = torch.tensor([[0.0]], dtype=torch.float32)
        val_ang = ang_loss(p_trig, t_ang)
        assert pytest.approx(val_ang.item(), 1e-4) == 0.0

        p_obb = torch.tensor([[100.0, 100.0, 40.0, 20.0, 0.0]], dtype=torch.float32)
        t_obb = torch.tensor([[100.0, 100.0, 40.0, 20.0, 0.0]], dtype=torch.float32)
        val_prob = prob_loss(p_obb, t_obb)
        assert pytest.approx(val_prob.item(), 1e-4) == 0.0
    else:
        assert ang_loss(None, None) == 0.0
        assert prob_loss(None, None) == 0.0
        total, bd = obb_loss({}, {})
        assert total == 0.0


# =========================================================================
# 4. Multi-Modal PredictionResult & Visualizer Tests
# =========================================================================

def test_prediction_result_multimodal_serialization(tmp_path):
    """Verify that PredictionResult stores and serializes pose, masks, and OBB."""
    boxes = np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32)
    scores = np.array([0.95], dtype=np.float32)
    classes = np.array([0], dtype=np.int64)
    labels = ["person"]

    # 17 keypoints
    kpts = np.zeros((1, 17, 3), dtype=np.float32)
    kpts[0, :, :2] = 50.0
    kpts[0, :, 2] = 0.99

    # OBB
    obb_angles = np.array([0.7854], dtype=np.float32)  # pi/4
    obb_corners = np.array([[[10.0, 20.0], [30.0, 20.0], [30.0, 50.0], [10.0, 50.0]]], dtype=np.float32)

    # Masks
    masks = np.ones((1, 32, 32), dtype=np.float32)

    res = PredictionResult(
        boxes=boxes,
        scores=scores,
        classes=classes,
        labels=labels,
        keypoints=kpts,
        masks=masks,
        obb_angles=obb_angles,
        obb_corners=obb_corners,
    )

    data = res.to_dict()
    assert data["num_detections"] == 1
    assert "keypoints" in data
    assert len(data["keypoints"][0]) == 17
    assert "obb_angles" in data
    assert "obb_corners" in data
    assert data["num_masks"] == 1

    # Save to disk
    json_path = tmp_path / "test_result.json"
    res.save_json(json_path)
    assert json_path.exists()


def test_visualizer_rendering():
    """Verify AurelixVisualizer renders pose skeletons, masks, and OBB polygons without error."""
    vis = AurelixVisualizer()
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)

    # 1. Pose rendering
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
    assert img_pose.size == (200, 200)

    # 2. Instance Segmentation rendering
    masks = np.zeros((1, 200, 200), dtype=np.float32)
    masks[0, 50:150, 50:150] = 1.0
    res_seg = PredictionResult(
        boxes=np.array([[50.0, 50.0, 150.0, 150.0]], dtype=np.float32),
        scores=np.array([0.88], dtype=np.float32),
        classes=np.array([1], dtype=np.int64),
        labels=["car"],
        masks=masks,
    )
    img_seg = vis.draw(canvas, res_seg)
    assert isinstance(img_seg, Image.Image)

    # 3. OBB rendering
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


def test_build_model_multimodal():
    """Verify build_model universal factory builds Pose, Segment, and OBB models."""
    from aurelix.models.aurelixnet import build_model

    m_pose = build_model("aurelix_pose_small")
    assert isinstance(m_pose, AurelixPoseModel)

    m_seg = build_model("aurelix_segment_small")
    assert isinstance(m_seg, AurelixSegmentModel)

    m_obb = build_model("aurelix_obb_small")
    assert isinstance(m_obb, AurelixOBBModel)


def test_multimodal_universal_export(tmp_path):
    """Verify UniversalExporter exports multi-modal models to target formats."""
    from aurelix.export.universal import UniversalExporter

    pose_model = build_aurelix_pose_small()
    exporter = UniversalExporter(model=pose_model)

    # Export to ONNX
    onnx_out = tmp_path / "pose.onnx"
    res_onnx = exporter.export("onnx", onnx_out)
    assert res_onnx["status"] in ["SUCCESS", "FALLBACK_SUCCESS"]
    assert onnx_out.exists()

    # Export to OpenVINO
    ov_out = tmp_path / "pose_openvino" / "model.xml"
    res_ov = exporter.export("openvino", ov_out)
    assert res_ov["status"] in ["SUCCESS", "FALLBACK_SUCCESS"]
    assert ov_out.exists()


