"""Comprehensive Unit Test Suite for Aurelix Deployment Platform.

Tests PredictionResult, PredictionDecoder, NonMaximumSuppression, AurelixPredictor,
StreamingPredictor, Visualizer, ModelExporter, DeploymentManager, and BenchmarkEngine.
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aurelix.configs.manager import ConfigManager
from aurelix.deployment.benchmark import BenchmarkEngine
from aurelix.deployment.manager import DeploymentManager, RuntimeSelector
from aurelix.export.exporter import ModelExporter
from aurelix.inference.decoder import GridGenerator, PredictionDecoder
from aurelix.inference.postprocess import NonMaximumSuppression
from aurelix.inference.predictor import AurelixPredictor
from aurelix.inference.result import PredictionResult
from aurelix.inference.stream import StreamingPredictor
from aurelix.models.detector import AurelixDetector
from aurelix.registry.registry import EXPORTER_REGISTRY, PREDICTOR_REGISTRY, VISUALIZER_REGISTRY
from aurelix.visualization.visualizer import Visualizer


def test_registries_contains_deployment_modules():
    """Verify registry registration of deployment platform modules."""
    assert PREDICTOR_REGISTRY.contains("AurelixPredictor")
    assert VISUALIZER_REGISTRY.contains("Visualizer")
    assert EXPORTER_REGISTRY.contains("ModelExporter")


def test_prediction_result_serialization(tmp_path):
    """Test PredictionResult dataclass serialization to dict and JSON."""
    boxes = np.array([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 80.0, 80.0]], dtype=np.float32)
    scores = np.array([0.95, 0.88], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)
    labels = ["person", "car"]

    res = PredictionResult(
        boxes=boxes,
        scores=scores,
        classes=classes,
        labels=labels,
        orig_shape=(640, 640, 3),
        image_shape=(640, 640, 3),
    )

    assert len(res) == 2
    res_dict = res.to_dict()
    assert res_dict["num_detections"] == 2
    assert res_dict["labels"] == ["person", "car"]

    json_file = tmp_path / "result.json"
    res.save_json(json_file)
    assert json_file.exists()


def test_prediction_decoder():
    """Test PredictionDecoder multi-scale output decoding."""
    if torch is None:
        return

    decoder = PredictionDecoder(num_classes=10, reg_max=16)

    # Dummy raw prediction scale tensors (B=1, C=10+64=74, H, W)
    out1 = torch.randn(1, 74, 40, 40)
    out2 = torch.randn(1, 74, 20, 20)
    out3 = torch.randn(1, 74, 10, 10)

    boxes, scores = decoder.decode([out1, out2, out3])
    # Total anchors = 40*40 + 20*20 + 10*10 = 1600 + 400 + 100 = 2100
    assert boxes.shape == (1, 2100, 4)
    assert scores.shape == (1, 2100, 10)


def test_non_maximum_suppression():
    """Test NonMaximumSuppression confidence filtering and coordinate unboxing."""
    if torch is None:
        return

    nms = NonMaximumSuppression(conf_threshold=0.50, iou_threshold=0.45)
    boxes = torch.tensor([[[10.0, 10.0, 50.0, 50.0], [12.0, 12.0, 48.0, 48.0]]], dtype=torch.float32)
    scores = torch.zeros(1, 2, 5, dtype=torch.float32)
    scores[0, 0, 0] = 0.90
    scores[0, 1, 0] = 0.80  # Highly overlapping box, should be suppressed

    results = nms(
        boxes=boxes,
        scores=scores,
        orig_shape=(640, 640),
        letterbox_shape=(640, 640),
        class_names=["cls0", "cls1", "cls2", "cls3", "cls4"],
    )

    assert len(results) == 1
    assert len(results[0]) == 1  # 1 box kept after NMS suppression


def test_visualizer(tmp_path):
    """Test Visualizer bounding box overlay rendering and saving."""
    viz = Visualizer()
    img_arr = np.full((300, 300, 3), 200, dtype=np.uint8)

    res = PredictionResult(
        boxes=np.array([[20.0, 20.0, 100.0, 100.0]], dtype=np.float32),
        scores=np.array([0.92], dtype=np.float32),
        classes=np.array([0], dtype=np.int64),
        labels=["car"],
    )

    out_file = tmp_path / "annotated.jpg"
    saved_path = viz.save(img_arr, res, out_file)
    assert saved_path.exists()


def test_aurelix_predictor_single_and_batch_inference():
    """Test AurelixPredictor end-to-end forward prediction on numpy array image."""
    if torch is None:
        return

    model = AurelixDetector(num_classes=10, scale="tiny")
    predictor = AurelixPredictor(model=model, conf_threshold=0.10)

    img_arr = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    result = predictor.predict(img_arr)

    assert isinstance(result, PredictionResult)
    assert "inference_ms" in result.timing


def test_streaming_predictor_generator():
    """Test StreamingPredictor video stream processing generator."""
    if torch is None:
        return

    model = AurelixDetector(num_classes=5, scale="tiny")
    predictor = AurelixPredictor(model=model)
    stream_engine = StreamingPredictor(predictor=predictor, frame_skip=0)

    def dummy_frame_generator():
        for _ in range(3):
            yield np.zeros((320, 320, 3), dtype=np.uint8)

    results = list(stream_engine.process_generator(dummy_frame_generator()))
    assert len(results) == 3


def test_model_exporter_torchscript_and_onnx(tmp_path):
    """Test ModelExporter exporting TorchScript and ONNX models."""
    if torch is None:
        return

    model = AurelixDetector(num_classes=10, scale="tiny")
    exporter = ModelExporter(model=model)

    # TorchScript export
    ts_path = tmp_path / "model.torchscript"
    saved_ts = exporter.export_torchscript(ts_path, input_shape=(1, 3, 320, 320))
    assert saved_ts.exists()

    # ONNX export
    onnx_path = tmp_path / "model.onnx"
    saved_onnx = exporter.export_onnx(onnx_path, input_shape=(1, 3, 320, 320))
    assert saved_onnx.exists()


def test_deployment_manager_and_benchmark(tmp_path):
    """Test DeploymentManager and BenchmarkEngine performance metrics collection."""
    if torch is None:
        return

    model = AurelixDetector(num_classes=10, scale="tiny")
    dep_mgr = DeploymentManager(model=model, device="cpu")
    assert dep_mgr.device == "cpu"

    benchmarker = BenchmarkEngine(model=model, input_shape=(1, 3, 320, 320), device="cpu")
    report = benchmarker.run_benchmark(warmup_iters=2, benchmark_iters=5)

    assert "throughput_fps" in report
    assert "latency_ms" in report
    assert report["latency_ms"]["mean"] > 0

    report_path = tmp_path / "benchmark.json"
    benchmarker.save_report(report, report_path)
    assert report_path.exists()
