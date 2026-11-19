"""Comprehensive Unit Test Suite for Aurelix AI Platform Ecosystem (v1.1).

Tests WeightManager, WeightVerifier, DatasetReporter, VisualReporter, Benchmark scripts,
and automated Model Packaging.
"""

import json
import pytest
from pathlib import Path

from scripts.benchmarks.benchmark_export import run_export_benchmark
from scripts.benchmarks.benchmark_inference import run_inference_benchmark
from scripts.benchmarks.benchmark_jetson import run_jetson_benchmark
from scripts.benchmarks.benchmark_latency import run_latency_benchmark
from scripts.benchmarks.benchmark_memory import run_memory_benchmark
from scripts.benchmarks.benchmark_scaling import run_scaling_benchmark
from scripts.package_model import package_model
from aurelix.utils.dataset_reporter import DatasetReporter
from aurelix.utils.weight_manager import WeightManager, WeightMetadata, WeightVerifier
from aurelix.visualization.reports import VisualReporter


def test_weight_manager_and_verifier(tmp_path):
    """Test WeightManager, WeightVerifier, and SHA256 checksums."""
    dummy_file = tmp_path / "model.pt"
    dummy_file.write_bytes(b"AURELIX_WEIGHTS_DUMMY_DATA_12345")

    sha256 = WeightVerifier.compute_sha256(dummy_file)
    assert len(sha256) == 64
    assert WeightVerifier.verify(dummy_file, sha256)

    mgr = WeightManager(weights_dir=tmp_path)
    meta = WeightMetadata(model_name="AurelixNet-Small", sha256=sha256)
    card_file = tmp_path / "metadata.json"
    mgr.create_model_card(meta, card_file)

    assert card_file.exists()
    loaded_meta = mgr.load_metadata(card_file)
    assert loaded_meta.model_name == "AurelixNet-Small"


def test_dataset_reporter(tmp_path):
    """Test DatasetReporter quality analytics and JSON export."""
    reporter = DatasetReporter(class_names=["person", "car"])
    anns = [
        {"class_id": 0, "bbox": [10, 10, 50, 50]},
        {"class_id": 0, "bbox": [20, 20, 80, 80]},
        {"class_id": 1, "bbox": [5, 5, 30, 40]},
    ]

    report = reporter.analyze(num_images=2, annotations=anns)
    assert report["num_images"] == 2
    assert report["total_objects"] == 3
    assert "person" in report["class_distribution"]
    assert report["dataset_quality_score"] > 0

    json_file = tmp_path / "dataset_report.json"
    reporter.save_json(report, json_file)
    assert json_file.exists()


def test_visual_reporter(tmp_path):
    """Test VisualReporter loss curve plotting."""
    plot_file = tmp_path / "loss_curves.png"
    history = {"train_loss": [1.5, 1.2, 0.9, 0.7], "val_loss": [1.6, 1.3, 1.0, 0.8]}
    saved_path = VisualReporter.plot_loss_curves(history, plot_file)
    assert saved_path.exists()


def test_benchmark_scripts(tmp_path):
    """Test platform benchmark suite execution."""
    mem_res = run_memory_benchmark()
    assert "tiny" in mem_res

    scale_res = run_scaling_benchmark()
    assert "aurelixnet_small" in scale_res

    jetson_res = run_jetson_benchmark()
    assert "aurelixnet_tiny" in jetson_res

    inf_res = run_inference_benchmark()
    assert "tiny" in inf_res

    lat_res = run_latency_benchmark()
    assert "tiny" in lat_res

    exp_res = run_export_benchmark(tmp_dir=str(tmp_path / "exports"))
    assert "tiny" in exp_res


def test_package_model_release(tmp_path):
    """Test automated model release packaging."""
    release_dir = tmp_path / "release_test"
    package_model(model_name="aurelixnet_tiny", num_classes=10, output_dir=release_dir)

    assert (release_dir / "model.pt").exists()
    assert (release_dir / "model.onnx").exists()
    assert (release_dir / "model.torchscript").exists()
    assert (release_dir / "metadata.json").exists()
    assert (release_dir / "checksums.json").exists()
    assert (release_dir / "README.md").exists()
