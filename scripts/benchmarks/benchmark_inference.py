"""Benchmark Inference Throughput and FPS across AurelixNet Model Family.

Measures inference FPS and end-to-end processing pipeline speed for AurelixNet variants.
"""

from typing import Dict

from aurelix.deployment.benchmark import BenchmarkEngine
from aurelix.models.aurelixnet import build_model
from aurelix.utils.logger import get_logger

logger = get_logger("aurelix.benchmarks.inference")


def run_inference_benchmark() -> Dict[str, float]:
    """Benchmark inference FPS for AurelixNet variants."""
    results = {}
    variants = ["tiny", "small", "medium", "large", "xlarge"]

    for var in variants:
        model = build_model(f"aurelixnet_{var}", num_classes=80)
        benchmarker = BenchmarkEngine(model=model, input_shape=(1, 3, 640, 640), device="cpu")
        report = benchmarker.run_benchmark(warmup_iters=5, benchmark_iters=15)
        fps = report.get("throughput_fps", 0.0)
        results[var] = fps
        logger.info(f"AurelixNet-{var.upper()} Inference FPS: {fps:.2f}")

    return results


if __name__ == "__main__":
    run_inference_benchmark()
