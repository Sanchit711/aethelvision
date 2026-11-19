"""Benchmark Detailed Latency Percentiles (Mean, Median, P95, P99) across AurelixNet Family.

Measures inference latency distributions across input resolutions.
"""

from typing import Dict

from aurelix.deployment.benchmark import BenchmarkEngine
from aurelix.models.aurelixnet import build_model
from aurelix.utils.logger import get_logger

logger = get_logger("aurelix.benchmarks.latency")


def run_latency_benchmark() -> Dict[str, Dict[str, float]]:
    """Benchmark latency percentiles for AurelixNet variants."""
    results = {}
    variants = ["tiny", "small", "medium", "large", "xlarge"]

    for var in variants:
        model = build_model(f"aurelixnet_{var}", num_classes=80)
        benchmarker = BenchmarkEngine(model=model, input_shape=(1, 3, 640, 640), device="cpu")
        report = benchmarker.run_benchmark(warmup_iters=5, benchmark_iters=15)
        results[var] = report.get("latency_ms", {})
        logger.info(f"AurelixNet-{var.upper()} Latency Percentiles: {results[var]}")

    return results


if __name__ == "__main__":
    run_latency_benchmark()
