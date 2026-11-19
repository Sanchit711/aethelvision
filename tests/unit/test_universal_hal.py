"""Unit Tests for Universal Hardware Abstraction Layer (HAL) & Multi-Backend Engines."""

import os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from aurelix.engine.hal import HardwareHAL
from aurelix.inference.engine import UniversalInferenceEngine
from aurelix.export.universal import UniversalExporter


class TestUniversalHardwareHAL:
    """Test suite for HardwareHAL."""

    def test_hardware_discovery(self):
        info = HardwareHAL.discover_hardware(force_refresh=True)
        assert isinstance(info, dict)
        assert "optimal_backend" in info
        assert "detected_backends" in info
        assert "backend_details" in info
        assert "cpu" in info["detected_backends"]
        assert info["optimal_backend"] in HardwareHAL.SUPPORTED_BACKENDS

    def test_get_optimal_device(self):
        device, backend = HardwareHAL.get_optimal_device()
        assert backend in HardwareHAL.SUPPORTED_BACKENDS
        assert device is not None

    def test_configure_runtime_optimizations(self):
        res = HardwareHAL.configure_runtime_optimizations()
        assert isinstance(res, dict)
        assert "backend" in res

    def test_empty_cache_all_backends(self):
        for backend in ["cuda", "mps", "xpu", "directml", "openvino", "cpu"]:
            # Must never raise an unhandled exception
            HardwareHAL.empty_cache(backend=backend)

    def test_amp_context(self):
        ctx, enabled = HardwareHAL.get_amp_context("cpu")
        assert ctx is not None
        assert isinstance(enabled, bool)
        with ctx:
            pass  # Ensure context enters and exits cleanly


class TestUniversalInferenceEngine:
    """Test suite for UniversalInferenceEngine."""

    def test_engine_initialization(self, tmp_path):
        dummy_model = tmp_path / "model.onnx"
        dummy_model.write_bytes(b"MOCK_ONNX_MODEL_CONTENT")

        engine = UniversalInferenceEngine(model_path=dummy_model, imgsz=(320, 320))
        assert engine.imgsz == (320, 320)
        assert engine.active_provider is not None

    def test_preprocess_pil_image(self, tmp_path):
        dummy_model = tmp_path / "model.onnx"
        dummy_model.write_bytes(b"MOCK_ONNX_MODEL_CONTENT")
        engine = UniversalInferenceEngine(model_path=dummy_model, imgsz=(640, 640))

        # Create a synthetic 800x600 RGB image
        img = Image.new("RGB", (800, 600), color=(255, 0, 0))
        batch, meta = engine.preprocess(img)

        assert isinstance(batch, np.ndarray)
        assert batch.shape == (1, 3, 640, 640)
        assert batch.dtype == np.float32
        assert meta["orig_size"] == (800, 600)
        assert "scale" in meta
        assert "pad" in meta

    def test_postprocess_detections(self, tmp_path):
        dummy_model = tmp_path / "model.onnx"
        dummy_model.write_bytes(b"MOCK_ONNX_MODEL_CONTENT")
        engine = UniversalInferenceEngine(model_path=dummy_model, conf_thresh=0.20)

        # Synthetic raw prediction tensor: 100 anchors, 4 box coords + 2 classes
        # Box centered at (320, 320, 100, 100) with high confidence for class 0
        raw = np.zeros((1, 6, 100), dtype=np.float32)
        raw[0, 0, 10] = 320.0  # x
        raw[0, 1, 10] = 320.0  # y
        raw[0, 2, 10] = 100.0  # w
        raw[0, 3, 10] = 100.0  # h
        raw[0, 4, 10] = 0.95   # class 0 confidence
        raw[0, 5, 10] = 0.05   # class 1 confidence

        meta = {
            "orig_size": (640, 640),
            "scale": 1.0,
            "pad": (0, 0),
            "new_size": (640, 640),
        }
        detections = engine.postprocess(raw, meta)
        assert isinstance(detections, list)
        assert len(detections) >= 1
        d = detections[0]
        assert d["class_id"] == 0
        assert d["score"] >= 0.90
        assert len(d["box"]) == 4

    def test_engine_predict_and_benchmark(self, tmp_path):
        dummy_model = tmp_path / "model.onnx"
        dummy_model.write_bytes(b"MOCK_ONNX_MODEL_CONTENT")
        engine = UniversalInferenceEngine(model_path=dummy_model, imgsz=(320, 320))

        img = Image.new("RGB", (320, 320), color=(100, 150, 200))
        res = engine.predict(img)
        assert "detections" in res
        assert "latency_ms" in res
        assert "fps" in res
        assert "provider" in res

        bench_res = engine.benchmark(iterations=5)
        assert "avg_latency_ms" in bench_res
        assert "fps" in bench_res
        assert bench_res["iterations"] == 5


class TestUniversalExporter:
    """Test suite for UniversalExporter multi-backend exports."""

    def test_export_onnx(self, tmp_path):
        exporter = UniversalExporter(model=None)
        out_p = tmp_path / "test.onnx"
        res = exporter.export("onnx", output_path=out_p)
        assert "SUCCESS" in res["status"]
        assert out_p.exists()

    def test_export_openvino(self, tmp_path):
        exporter = UniversalExporter(model=None)
        out_p = tmp_path / "test.xml"
        res = exporter.export("openvino", output_path=out_p)
        assert "SUCCESS" in res["status"]
        assert res["format"] == "OpenVINO IR"
        assert Path(res["xml_path"]).exists()

    def test_export_coreml(self, tmp_path):
        exporter = UniversalExporter(model=None)
        out_p = tmp_path / "test.mlpackage"
        res = exporter.export("coreml", output_path=out_p)
        assert "SUCCESS" in res["status"]
        assert res["format"] == "CoreML"
        assert Path(res["path"]).exists()

    def test_export_tensorrt(self, tmp_path):
        exporter = UniversalExporter(model=None)
        out_p = tmp_path / "test.engine"
        res = exporter.export("tensorrt", output_path=out_p)
        assert "SUCCESS" in res["status"]
        assert Path(res["path"]).exists()

    def test_export_torchscript(self, tmp_path):
        exporter = UniversalExporter(model=None)
        out_p = tmp_path / "test.torchscript"
        res = exporter.export("torchscript", output_path=out_p)
        assert "SUCCESS" in res["status"]
        assert Path(res["path"]).exists()

    def test_export_invalid_format(self, tmp_path):
        exporter = UniversalExporter(model=None)
        with pytest.raises(ValueError):
            exporter.export("invalid_format_xyz", output_path=tmp_path / "test.bin")


class TestAurelixTransformer:
    """Test suite for AurelixTransformer NMS-Free Vision Transformer."""

    def test_transformer_init(self):
        from aurelix.models.transformer import (
            AurelixTransformer,
            build_transformer_base,
            build_transformer_small,
        )

        model = AurelixTransformer(num_classes=20, num_queries=100)
        assert model.num_classes == 20
        assert model.num_queries == 100

        small = build_transformer_small(num_classes=80)
        assert small.num_queries == 100

        base = build_transformer_base(num_classes=80)
        assert base.num_queries == 300

    def test_transformer_forward_mock(self):
        from aurelix.models.transformer import AurelixTransformer

        model = AurelixTransformer(num_classes=10, num_queries=50)
        dummy_input = None
        out = model(dummy_input)
        assert isinstance(out, dict)
        assert out.get("nms_required") is False
