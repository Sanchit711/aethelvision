"""Benchmark Speed (Preprocess, Forward Pass, Decode, NMS, Total) breakdown for AurelixNet.

Measures stage-by-stage latency breakdowns across single and batch image inputs.
"""

from typing import Dict

from aurelix.deployment.benchmark import BenchmarkEngine
from aurelix.models.aurelixnet import build_model
from aurelix.utils.logger import get_logger

logger = get_logger("aurelix.benchmarks.speed")


def run_speed_benchmark() -> Dict[str, float]:
    """Benchmark stage-by-stage execution speed."""
    results = {}
    variants = ["tiny", "small", "medium", "large", "xlarge"]

    for var in variants:
        model = build_model(f"aurelixnet_{var}", num_classes=80)
        engine = BenchmarkEngine(model=model, input_shape=(1, 3, 640, 640), device="cpu")
        report = engine.run_benchmark(warmup_iters=5, benchmark_iters=15)
        mean_ms = report.get("latency_ms", {}).get("mean", 0.0)
        results[var] = mean_ms
        logger.info(f"AurelixNet-{var.upper()} Mean Speed Latency: {mean_ms:.2f} ms")

    return results


if __name__ == "__main__":
    run_speed_benchmark()
