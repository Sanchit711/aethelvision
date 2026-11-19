"""Unit tests for Quantization and Multi-Camera Streaming subsystems."""

import pytest
from pathlib import Path
import tempfile

from aethel.export.quantize import quantize_onnx_dynamic, quantize_torch_model
from stream import VideoStreamWorker, MultiCameraStream


class TestQuantizationSubsystem:
    """Tests INT8 Post-Training Quantization module."""

    def test_quantize_torch_model(self):
        try:
            import torch
            import torch.nn as nn
            model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
            quantized = quantize_torch_model(model)
            assert quantized is not None
        except ImportError:
            pytest.skip("Torch not installed")

    def test_quantize_onnx_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_onnx = Path(tmpdir) / "test_model.onnx"
            dummy_onnx.write_bytes(b"MOCK_ONNX_MODEL_BINARY_FOR_TESTING_1234567890")
            
            out_onnx = quantize_onnx_dynamic(dummy_onnx)
            assert out_onnx.exists()
            assert out_onnx.stat().st_size > 0


class TestStreamingSubsystem:
    """Tests multi-camera ring buffer and streaming workers."""

    def test_mock_video_stream_worker(self):
        worker = VideoStreamWorker(source="mock", stream_id=0, buffer_size=2)
        worker.start()
        
        # Read a frame
        frame = worker.read(timeout=1.0)
        assert frame is not None
        assert hasattr(frame, "shape")
        assert frame.shape == (480, 640, 3)
        
        worker.stop()
        assert not worker.running

    def test_multicamera_context_manager(self):
        with MultiCameraStream(sources=["mock", "mock"]) as streamer:
            frames = streamer.read_all(timeout=1.0)
            assert len(frames) == 2
            assert frames[0] is not None
            assert frames[1] is not None
