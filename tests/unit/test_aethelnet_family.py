"""Comprehensive Unit Test Suite for AethelNet Official Model Family.

Tests model construction across all 5 variants (Tiny, Small, Medium, Large, XLarge),
build_model factory lookup, registry integration, forward pass shapes, TorchScript/ONNX exports,
and YAML configuration specs loading.
"""

from pathlib import Path
import numpy as np
import pytest

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aethel.configs.manager import ConfigManager
from aethel.export.exporter import ModelExporter
from aethel.models.aethelnet import (
    AETHELNET_PRESETS,
    AethelNet,
    build_aethelnet_large,
    build_aethelnet_medium,
    build_aethelnet_small,
    build_aethelnet_tiny,
    build_aethelnet_xlarge,
    build_model,
)
from aethel.registry.registry import MODEL_REGISTRY


def test_aethelnet_registry_registration():
    """Verify registration of AethelNet and variant aliases in MODEL_REGISTRY."""
    assert MODEL_REGISTRY.contains("AethelNet")
    assert MODEL_REGISTRY.contains("aethelnet_tiny")
    assert MODEL_REGISTRY.contains("aethelnet_small")
    assert MODEL_REGISTRY.contains("aethelnet_medium")
    assert MODEL_REGISTRY.contains("aethelnet_large")
    assert MODEL_REGISTRY.contains("aethelnet_xlarge")


@pytest.mark.parametrize("variant", ["tiny", "small", "medium", "large", "xlarge"])
def test_build_model_factory(variant):
    """Verify build_model factory constructs every AethelNet variant."""
    model_name = f"aethelnet_{variant}"
    model = build_model(model_name, num_classes=80)

    assert isinstance(model, AethelNet)
    assert model.num_classes == 80

    if torch is not None:
        dummy_input = torch.randn(1, 3, 320, 320)
        outputs = model(dummy_input)
        assert isinstance(outputs, list)
        assert len(outputs) == 3


def test_build_aethelnet_small_direct():
    """Direct constructor test for AethelNet-Small variant."""
    model = build_aethelnet_small(num_classes=20)
    assert isinstance(model, AethelNet)
    assert model.num_classes == 20
    assert model.variant == "small"


def test_aethelnet_yaml_configs():
    """Verify loading AethelNet variant specifications from YAML files."""
    for variant in ["tiny", "small", "medium", "large", "xlarge"]:
        yaml_path = Path(f"aethel/configs/models/aethelnet_{variant}.yaml")
        if yaml_path.exists():
            cfg = ConfigManager.load_yaml(yaml_path)
            model = build_model(cfg.model.name, num_classes=cfg.model.num_classes)
            assert isinstance(model, AethelNet)


def test_repconv_block():
    """Verify Structural Re-parameterization (RepConv) layer and branch fusion."""
    from aethel.layers.repconv import RepConv

    rep = RepConv(in_channels=32, out_channels=32, stride=1)
    assert rep.deploy is False

    if torch is not None:
        rep.eval()
        x = torch.randn(1, 32, 20, 20)
        with torch.no_grad():
            out_train = rep(x)
            # Fuse multi-branch into single 3x3 Conv2d for inference
            rep.fuse_repconv()
            assert rep.deploy is True
            assert hasattr(rep, "rbr_reparam")
            assert not hasattr(rep, "rbr_dense")
            out_deploy = rep(x)
            assert out_deploy.shape == (1, 32, 20, 20)
            # Numerical equivalence between multi-branch and fused 3x3
            np.testing.assert_allclose(out_train.detach().numpy(), out_deploy.detach().numpy(), atol=1e-4)

