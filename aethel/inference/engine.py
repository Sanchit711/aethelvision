"""Universal Hardware-Agnostic Inference Engine for AethelVision / AETHEL.

Dynamically binds and executes on any hardware without requiring an NVIDIA GPU:
- Apple Silicon (via CoreMLExecutionProvider on Apple Neural Engine)
- AMD GPUs (via DmlExecutionProvider on Windows, ROCm on Linux)
- Intel Accelerators (via OpenVINOExecutionProvider on Core Ultra / Arc)
- Qualcomm Snapdragon (via QNNExecutionProvider)
- NVIDIA GPUs (via CUDAExecutionProvider / TensorrtExecutionProvider)
- Universal Multi-core CPUs (via CPUExecutionProvider with AVX2/AVX-512/NEON)
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore
    ORT_AVAILABLE = False

from aethel.engine.hal import HardwareHAL
from aethel.utils.logger import get_logger

logger = get_logger("aethel.inference.universal_engine")


class UniversalInferenceEngine:
    """Universal High-Throughput Inference Engine.
    
    Automatically discovers and binds to the optimal execution provider
    installed on the host hardware.
    """

    PROVIDER_PRIORITY = [
        ("CoreMLExecutionProvider", {"enable_on_subgraph": True}),
        ("TensorrtExecutionProvider", {"device_id": 0}),
        ("CUDAExecutionProvider", {"device_id": 0}),
        ("ROCMExecutionProvider", {"device_id": 0}),
        ("DmlExecutionProvider", {"device_id": 0}),
        ("OpenVINOExecutionProvider", {"device_type": "CPU_FP32"}),
        ("QNNExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ]

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        imgsz: Tuple[int, int] = (640, 640),
        preferred_provider: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.preferred_provider = preferred_provider

        self.session: Optional[Any] = None
        self.active_provider: str = "Mock/CPU"
        self.input_name: str = "images"
        self.input_shape: List[Any] = [1, 3, imgsz[0], imgsz[1]]

        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        """Initialize the optimal execution session."""
        if not ORT_AVAILABLE:
            logger.warning(
                "onnxruntime is not installed. UniversalInferenceEngine running in simulated mock mode."
            )
            self.active_provider = "Simulated Fallback (Install onnxruntime for hardware acceleration)"
            return

        if not self.model_path.exists():
            logger.warning(f"Model path '{self.model_path}' not found on disk. Initializing in standby mode.")
            self.active_provider = "Standby (Awaiting Weights)"
            return

        available = ort.get_available_providers()
        chosen_providers = []

        if self.preferred_provider and self.preferred_provider in available:
            chosen_providers.append(self.preferred_provider)

        for prov_name, prov_opts in self.PROVIDER_PRIORITY:
            if prov_name in available and prov_name not in chosen_providers:
                chosen_providers.append((prov_name, prov_opts))

        # Set aggressive graph optimizations
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.intra_op_num_threads = 0  # auto-tune to hardware cores

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_opts,
                providers=chosen_providers,
            )
            self.active_provider = self.session.get_providers()[0]
            inputs = self.session.get_inputs()
            if inputs:
                self.input_name = inputs[0].name
                self.input_shape = inputs[0].shape
            logger.info(f"Initialized UniversalInferenceEngine on backend: {self.active_provider}")
        except Exception as e:
            logger.warning(f"Failed to load ONNX session with providers {chosen_providers}: {e}. Falling back to CPU.")
            try:
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self.active_provider = "CPUExecutionProvider"
            except Exception as cpu_e:
                logger.error(f"CPU fallback also failed: {cpu_e}")
                self.active_provider = f"Failed ({cpu_e})"

    def preprocess(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess input image to normalized CHW float32 tensor."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        orig_w, orig_h = pil_img.size
        target_h, target_w = self.imgsz

        # Letterbox resize maintaining aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Pad to target dimensions
        canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))

        # Convert to float32 CHW array
        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
        batch = np.expand_dims(arr, axis=0)  # Add batch dimension

        meta = {
            "orig_size": (orig_w, orig_h),
            "scale": scale,
            "pad": (pad_x, pad_y),
            "new_size": (new_w, new_h),
        }
        return batch, meta

    def postprocess(
        self,
        raw_outputs: np.ndarray,
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Decode raw prediction tensor into scaled bounding box detections."""
        detections: List[Dict[str, Any]] = []

        if raw_outputs is None or len(raw_outputs) == 0:
            return detections

        # Squeeze batch dimension if present
        preds = raw_outputs[0] if raw_outputs.ndim == 3 else raw_outputs

        # Transpose if output shape is (features, anchors) e.g., (84, 8400) -> (8400, 84)
        if preds.shape[0] < preds.shape[1] and preds.shape[0] in (5, 6, 84, 85):
            preds = np.transpose(preds, (1, 0))

        # If predictions format is [x, y, w, h, class_scores...]
        if preds.shape[1] >= 5:
            boxes = preds[:, :4]
            scores = preds[:, 4:]
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)

            # Filter by confidence threshold
            mask = max_scores >= self.conf_thresh
            boxes = boxes[mask]
            max_scores = max_scores[mask]
            class_ids = class_ids[mask]

            pad_x, pad_y = meta["pad"]
            scale = meta["scale"]
            orig_w, orig_h = meta["orig_size"]

            for box, score, cid in zip(boxes, max_scores, class_ids):
                # Unpad and scale back to original image dimensions
                x1 = max(0, min(orig_w, (box[0] - box[2] / 2 - pad_x) / scale))
                y1 = max(0, min(orig_h, (box[1] - box[3] / 2 - pad_y) / scale))
                x2 = max(0, min(orig_w, (box[0] + box[2] / 2 - pad_x) / scale))
                y2 = max(0, min(orig_h, (box[1] + box[3] / 2 - pad_y) / scale))

                detections.append({
                    "box": [round(float(v), 2) for v in (x1, y1, x2, y2)],
                    "score": round(float(score), 4),
                    "class_id": int(cid),
                })

        return detections

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """Execute full end-to-end detection on a single image."""
        t0 = time.perf_counter()
        batch, meta = self.preprocess(image)
        t_prep = time.perf_counter()

        if self.session is not None:
            raw_out = self.session.run(None, {self.input_name: batch})[0]
        else:
            # Simulated output when running in mock mode
            raw_out = np.zeros((1, 84, 8400), dtype=np.float32)

        t_infer = time.perf_counter()
        detections = self.postprocess(raw_out, meta)
        t_post = time.perf_counter()

        return {
            "detections": detections,
            "provider": self.active_provider,
            "latency_ms": {
                "preprocess": round((t_prep - t0) * 1000, 2),
                "inference": round((t_infer - t_prep) * 1000, 2),
                "postprocess": round((t_post - t_infer) * 1000, 2),
                "total": round((t_post - t0) * 1000, 2),
            },
            "fps": round(1.0 / max(1e-5, (t_post - t0)), 1),
        }

    def benchmark(self, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark latency and throughput on the active hardware accelerator."""
        dummy_input = np.random.randn(1, 3, self.imgsz[0], self.imgsz[1]).astype(np.float32)

        if self.session is None:
            return {
                "status": "SIMULATED",
                "provider": self.active_provider,
                "iterations": iterations,
                "avg_latency_ms": 7.05,
                "min_latency_ms": 6.80,
                "max_latency_ms": 8.10,
                "fps": 141.8,
            }

        # Warmup
        for _ in range(5):
            _ = self.session.run(None, {self.input_name: dummy_input})

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = self.session.run(None, {self.input_name: dummy_input})
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        avg_lat = sum(latencies) / len(latencies)
        fps = 1000.0 / avg_lat

        return {
            "status": "ACTIVE",
            "provider": self.active_provider,
            "iterations": iterations,
            "avg_latency_ms": round(avg_lat, 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "fps": round(fps, 1),
        }


__all__ = ["UniversalInferenceEngine"]
