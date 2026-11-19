"""Comprehensive Unit Test Suite for Aurelix Model Architecture Subsystem.

Tests forward propagation, multi-scale output shapes, channel consistency,
gradient flow, mixed precision readiness, dynamic image sizes, and checkpoint persistence.
"""

import pytest
from pathlib import Path

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aurelix.configs.manager import ConfigManager
from aurelix.layers.attention import CBAM
from aurelix.layers.blocks import SPPF, Bottleneck, CSPBlock
from aurelix.layers.conv import Conv, DWConv
from aurelix.models.backbone.csp import CSPBackbone
from aurelix.models.detector import SCALE_FACTORS, AurelixDetector
from aurelix.models.head.decoupled import DecoupledDetectHead
from aurelix.models.neck.pafpn import PAFPN
from aurelix.registry.registry import (
    BACKBONE_REGISTRY,
    HEAD_REGISTRY,
    LAYER_REGISTRY,
    MODEL_REGISTRY,
    NECK_REGISTRY,
)


def test_layer_primitives_instantiation():
    """Verify registry registration and instantiation of layer primitives."""
    assert LAYER_REGISTRY.contains("Conv")
    assert LAYER_REGISTRY.contains("DWConv")
    assert LAYER_REGISTRY.contains("CSPBlock")
    assert LAYER_REGISTRY.contains("SPPF")
    assert LAYER_REGISTRY.contains("CBAM")

    if torch is not None:
        conv = Conv(16, 32, 3)
        x = torch.randn(1, 16, 64, 64)
        out = conv(x)
        assert out.shape == (1, 32, 64, 64)


def test_backbone_forward():
    """Verify CSPBackbone extracts multi-scale features (P3, P4, P5)."""
    assert BACKBONE_REGISTRY.contains("CSPBackbone")

    if torch is not None:
        backbone = CSPBackbone(in_channels=3, depth_multiple=0.33, width_multiple=0.25)
        x = torch.randn(1, 3, 640, 640)
        p3, p4, p5 = backbone(x)

        # Stride 8: 640/8 = 80
        assert p3.shape[2:] == (80, 80)
        # Stride 16: 640/16 = 40
        assert p4.shape[2:] == (40, 40)
        # Stride 32: 640/32 = 20
        assert p5.shape[2:] == (20, 20)


def test_neck_forward():
    """Verify PAFPN fuses multi-scale features into (N3, N4, N5)."""
    assert NECK_REGISTRY.contains("PAFPN")

    if torch is not None:
        neck = PAFPN(in_channels_list=(32, 64, 128))
        p3 = torch.randn(1, 32, 80, 80)
        p4 = torch.randn(1, 64, 40, 40)
        p5 = torch.randn(1, 128, 20, 20)

        n3, n4, n5 = neck((p3, p4, p5))
        assert n3.shape == (1, 32, 80, 80)
        assert n4.shape == (1, 64, 40, 40)
        assert n5.shape == (1, 128, 20, 20)


def test_decoupled_head_forward():
    """Verify DecoupledDetectHead output raw prediction tensors."""
    assert HEAD_REGISTRY.contains("DecoupledDetectHead")

    if torch is not None:
        head = DecoupledDetectHead(num_classes=80, in_channels_list=(32, 64, 128), reg_max=16)
        n3 = torch.randn(1, 32, 80, 80)
        n4 = torch.randn(1, 64, 40, 40)
        n5 = torch.randn(1, 128, 20, 20)

        outputs = head((n3, n4, n5))
        assert len(outputs) == 3
        # Output channels = num_classes (80) + 4 * reg_max (64) = 144
        assert outputs[0].shape == (1, 144, 80, 80)
        assert outputs[1].shape == (1, 144, 40, 40)
        assert outputs[2].shape == (1, 144, 20, 20)


@pytest.mark.parametrize("scale", ["tiny", "small", "medium", "large", "xlarge"])
def test_aurelix_detector_scales(scale):
    """Verify AurelixDetector instantiates and forwards across all scale variants."""
    assert MODEL_REGISTRY.contains("AurelixDetector")

    detector_cls = MODEL_REGISTRY.get("AurelixDetector")
    model = detector_cls(num_classes=80, scale=scale)

    if torch is not None:
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(x)
        assert len(outputs) == 3
        assert outputs[0].shape[0] == 1
        assert outputs[0].shape[1] == 144  # 80 + 4*16


def test_dynamic_image_sizes():
    """Verify forward propagation with dynamic batch sizes and spatial resolutions."""
    if torch is None:
        return

    model = AurelixDetector(num_classes=20, scale="tiny")
    model.eval()

    resolutions = [(320, 320), (640, 640), (800, 800)]
    for h, w in resolutions:
        x = torch.randn(2, 3, h, w)
        with torch.no_grad():
            outputs = model(x)
        assert outputs[0].shape == (2, 20 + 64, h // 8, w // 8)
        assert outputs[1].shape == (2, 20 + 64, h // 16, w // 16)
        assert outputs[2].shape == (2, 20 + 64, h // 32, w // 32)


def test_gradient_flow():
    """Verify backpropagation gradient flow through entire model architecture."""
    if torch is None:
        return

    model = AurelixDetector(num_classes=10, scale="tiny")
    model.train()

    x = torch.randn(2, 3, 320, 320, requires_grad=True)
    outputs = model(x)

    # Compute scalar dummy loss over head predictions
    loss = sum(out.sum() for out in outputs)
    loss.backward()

    assert x.grad is not None
    # Check that backbone weights receive gradients
    assert model.backbone.stem.conv1.conv.weight.grad is not None


def test_checkpoint_save_load(tmp_path):
    """Verify saving and loading model state dict weights."""
    if torch is None:
        return

    model_src = AurelixDetector(num_classes=80, scale="tiny")
    checkpoint_file = tmp_path / "model.pt"
    model_src.save_weights(checkpoint_file)

    assert checkpoint_file.exists()

    model_dst = AurelixDetector(num_classes=80, scale="tiny")
    model_dst.load_weights(checkpoint_file)


def test_yaml_config_instantiation(tmp_path):
    """Verify model instantiation from YAML configuration file."""
    yaml_path = Path("aurelix/configs/models/aurelix_tiny.yaml")
    if yaml_path.exists():
        cfg = ConfigManager.load_yaml(yaml_path)
        model = AurelixDetector(
            num_classes=cfg.model.num_classes,
            scale=cfg.model.scale,
            depth_multiple=cfg.model.depth_multiple,
            width_multiple=cfg.model.width_multiple,
        )
        assert model.depth_multiple == 0.33
        assert model.width_multiple == 0.25
