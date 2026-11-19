"""Comprehensive Unit Test Suite for AurelixNet Official Model Family.

Tests model construction across all 5 variants (Tiny, Small, Medium, Large, XLarge),
build_model factory lookup, registry integration, forward pass shapes, TorchScript/ONNX exports,
and YAML configuration specs loading.
"""

import pytest
from pathlib import Path

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aurelix.configs.manager import ConfigManager
from aurelix.export.exporter import ModelExporter
from aurelix.models.aurelixnet import (
    AURELIXNET_PRESETS,
    AurelixNet,
    build_aurelixnet_large,
    build_aurelixnet_medium,
    build_aurelixnet_small,
    build_aurelixnet_tiny,
    build_aurelixnet_xlarge,
    build_model,
)
from aurelix.registry.registry import MODEL_REGISTRY


def test_aurelixnet_registry_registration():
    """Verify registration of AurelixNet and variant aliases in MODEL_REGISTRY."""
    assert MODEL_REGISTRY.contains("AurelixNet")
    assert MODEL_REGISTRY.contains("aurelixnet_tiny")
    assert MODEL_REGISTRY.contains("aurelixnet_small")
    assert MODEL_REGISTRY.contains("aurelixnet_medium")
    assert MODEL_REGISTRY.contains("aurelixnet_large")
    assert MODEL_REGISTRY.contains("aurelixnet_xlarge")


@pytest.mark.parametrize("variant", ["tiny", "small", "medium", "large", "xlarge"])
def test_build_model_factory(variant):
    """Verify build_model factory constructs every AurelixNet variant."""
    model_name = f"aurelixnet_{variant}"
    model = build_model(model_name, num_classes=80)

    assert isinstance(model, AurelixNet)
    assert model.num_classes == 80

    if torch is not None:
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            outputs = model(x)
        assert len(outputs) == 3
        # Output shape for 320x320: Scale 1 is 320/8 = 40
        assert outputs[0].shape == (1, 144, 40, 40)


def test_aurelixnet_dynamic_resolutions():
    """Verify forward propagation across dynamic image resolutions."""
    if torch is None:
        return

    model = build_model("aurelixnet_small", num_classes=10)
    model.eval()

    for h, w in [(256, 256), (512, 512), (640, 640)]:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            outputs = model(x)
        assert outputs[0].shape == (1, 10 + 64, h // 8, w // 8)
        assert outputs[1].shape == (1, 10 + 64, h // 16, w // 16)
        assert outputs[2].shape == (1, 10 + 64, h // 32, w // 32)


def test_aurelixnet_yaml_configs():
    """Verify loading AurelixNet variant specifications from YAML files."""
    for variant in ["tiny", "small", "medium", "large", "xlarge"]:
        yaml_path = Path(f"aurelix/configs/models/aurelixnet_{variant}.yaml")
        if yaml_path.exists():
            cfg = ConfigManager.load_yaml(yaml_path)
            model = build_model(cfg.model.name, num_classes=cfg.model.num_classes)
            assert isinstance(model, AurelixNet)


def test_aurelixnet_export(tmp_path):
    """Verify TorchScript and ONNX export for AurelixNet models."""
    if torch is None:
        return

    model = build_model("aurelixnet_tiny", num_classes=80)
    exporter = ModelExporter(model=model)

    # ONNX export
    onnx_file = tmp_path / "aurelixnet_tiny.onnx"
    saved_onnx = exporter.export_onnx(onnx_file, input_shape=(1, 3, 320, 320))
    assert saved_onnx.exists()

    # TorchScript export
    ts_file = tmp_path / "aurelixnet_tiny.torchscript"
    saved_ts = exporter.export_torchscript(ts_file, input_shape=(1, 3, 320, 320))
    assert saved_ts.exists()
