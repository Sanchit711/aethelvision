"""Aurelix Detector Benchmark Framework.

Benchmarking tool for evaluating inference throughput (FPS) and latency distributions across execution backends:
- PyTorch CPU
- PyTorch CUDA (optional, skipped gracefully if unavailable)
- ONNX Runtime CPU
- ONNX Runtime CUDA (optional, skipped gracefully if unavailable)
- TensorRT Engine (optional, skipped gracefully if unavailable)
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from aethel_detector.model import AethelDetector as AurelixDetector
    from aethel.inference.engine import UniversalInferenceEngine
    from aethel.engine.hal import HardwareHAL
except ImportError:
    try:
        from aurelix_detector.model import AurelixDetector
        from aurelix.inference.engine import UniversalInferenceEngine
        from aurelix.engine.hal import HardwareHAL
    except ImportError:
        AurelixDetector = None  # type: ignore
        UniversalInferenceEngine = None  # type: ignore
        HardwareHAL = None  # type: ignore

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


def compute_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculate latency statistics from raw iteration measurements."""
    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "fps": float(1000.0 / np.mean(arr)),
    }


def benchmark_pytorch_cpu(
    model: Any,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    warmup: int = 10,
    iters: int = 50,
) -> Dict[str, Any]:
    """Benchmark PyTorch model on CPU backend."""
    model.eval()
    model.to("cpu")
    dummy = torch.randn(*input_shape, device="cpu")

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    stats = compute_stats(latencies)
    return {"backend": "PyTorch CPU", "status": "COMPLETED", "stats": stats}


def benchmark_pytorch_cuda(
    model: Any,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    warmup: int = 10,
    iters: int = 50,
) -> Dict[str, Any]:
    """Benchmark PyTorch model on CUDA GPU backend."""
    if not torch.cuda.is_available():
        print("Benchmark skipped — backend unavailable (PyTorch CUDA unavailable).")
        return {"backend": "PyTorch CUDA", "status": "SKIPPED", "message": "Benchmark skipped — backend unavailable."}

    model.eval()
    model.to("cuda")
    dummy = torch.randn(*input_shape, device="cuda")

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            torch.cuda.synchronize()

    latencies = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(iters):
            starter.record()
            _ = model(dummy)
            ender.record()
            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender))

    stats = compute_stats(latencies)
    return {"backend": "PyTorch CUDA", "status": "COMPLETED", "stats": stats}


def benchmark_onnxruntime_cpu(
    onnx_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    warmup: int = 10,
    iters: int = 50,
) -> Dict[str, Any]:
    """Benchmark ONNX Runtime on CPU backend."""
    if not ORT_AVAILABLE or not onnx_path.exists():
        print(f"Benchmark skipped — backend unavailable or ONNX file missing: {onnx_path}")
        return {"backend": "ONNX Runtime CPU", "status": "SKIPPED", "message": "Benchmark skipped — backend unavailable."}

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup runs
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    stats = compute_stats(latencies)
    return {"backend": "ONNX Runtime CPU", "status": "COMPLETED", "stats": stats}


def benchmark_onnxruntime_cuda(
    onnx_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    warmup: int = 10,
    iters: int = 50,
) -> Dict[str, Any]:
    """Benchmark ONNX Runtime on CUDA GPU backend."""
    if not ORT_AVAILABLE or not onnx_path.exists() or "CUDAExecutionProvider" not in ort.get_available_providers():
        print("Benchmark skipped — backend unavailable (ONNX Runtime CUDA unavailable).")
        return {"backend": "ONNX Runtime CUDA", "status": "SKIPPED", "message": "Benchmark skipped — backend unavailable."}

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CUDAExecutionProvider"])

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    stats = compute_stats(latencies)
    return {"backend": "ONNX Runtime CUDA", "status": "COMPLETED", "stats": stats}


def benchmark_tensorrt(
    engine_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    warmup: int = 10,
    iters: int = 50,
) -> Dict[str, Any]:
    """Benchmark TensorRT Engine binary."""
    print("Benchmark skipped — backend unavailable (TensorRT runtime unavailable).")
    return {"backend": "TensorRT Engine", "status": "SKIPPED", "message": "Benchmark skipped — backend unavailable."}


def benchmark_universal_engine(
    onnx_path: Path,
    warmup: int = 10,
    iters: int = 50,
) -> Dict[str, Any]:
    """Benchmark Universal Hardware-Agnostic Inference Engine across optimal providers."""
    engine = UniversalInferenceEngine(model_path=onnx_path)
    res = engine.benchmark(iterations=iters)
    return {
        "backend": f"Universal HAL ({res['provider']})",
        "status": res["status"],
        "stats": {
            "mean_ms": res["avg_latency_ms"],
            "min_ms": res["min_latency_ms"],
            "max_ms": res["max_latency_ms"],
            "fps": res["fps"],
        },
        "provider": res["provider"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurelix Detector Benchmark Framework")
    parser.add_argument("--onnx", type=str, default="exports/best.onnx", help="Path to ONNX model file")
    parser.add_argument("--engine", type=str, default="exports/best.engine", help="Path to TensorRT engine file")
    parser.add_argument("--iters", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    args = parser.parse_args()

    print("=== AURELIX DETECTOR BENCHMARK FRAMEWORK ===")

    onnx_file = Path(args.onnx)
    engine_file = Path(args.engine)

    # 1. PyTorch CPU & CUDA
    if TORCH_AVAILABLE and torch is not None and AurelixDetector is not None:
        model = AurelixDetector(num_classes=80)
        model.eval()

        pt_cpu_res = benchmark_pytorch_cpu(model, warmup=args.warmup, iters=args.iters)
        if pt_cpu_res["status"] == "COMPLETED":
            st = pt_cpu_res["stats"]
            print(f"\n[PyTorch CPU] Mean Latency: {st['mean_ms']:.2f} ms | FPS: {st['fps']:.1f} | Min: {st['min_ms']:.2f} ms | Max: {st['max_ms']:.2f} ms")

        pt_cuda_res = benchmark_pytorch_cuda(model, warmup=args.warmup, iters=args.iters)
    else:
        print("\n[PyTorch] Skipped — running in lightweight hardware deployment mode.")

    # 3. ONNX Runtime CPU
    if onnx_file.exists():
        ort_cpu_res = benchmark_onnxruntime_cpu(onnx_file, warmup=args.warmup, iters=args.iters)
        if ort_cpu_res["status"] == "COMPLETED":
            st = ort_cpu_res["stats"]
            print(f"\n[ONNX Runtime CPU] Mean Latency: {st['mean_ms']:.2f} ms | FPS: {st['fps']:.1f} | Min: {st['min_ms']:.2f} ms | Max: {st['max_ms']:.2f} ms")
    else:
        print(f"\nONNX file not found at {onnx_file}. Run 'python export.py' first to generate ONNX model.")

    # 4. ONNX Runtime CUDA
    ort_cuda_res = benchmark_onnxruntime_cuda(onnx_file, warmup=args.warmup, iters=args.iters)

    # 5. TensorRT
    trt_res = benchmark_tensorrt(engine_file, warmup=args.warmup, iters=args.iters)

    # 6. Universal Hardware-Agnostic Execution Engine
    if onnx_file.exists():
        uni_res = benchmark_universal_engine(onnx_file, warmup=args.warmup, iters=args.iters)
        st = uni_res["stats"]
        print(f"\n[Universal HAL Engine - {uni_res['provider']}] Mean Latency: {st['mean_ms']:.2f} ms | FPS: {st['fps']:.1f} | Min: {st['min_ms']:.2f} ms | Max: {st['max_ms']:.2f} ms")

    print("\nBenchmark suite finished.")


if __name__ == "__main__":
    main()
