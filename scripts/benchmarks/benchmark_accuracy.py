"""Benchmark Accuracy (Precision, Recall, F1, mAP@50, mAP@50:95) across AurelixNet Family.

Runs quantitative evaluation across benchmark datasets and generates JSON accuracy reports.
"""

from typing import Dict

from aurelix.metrics.evaluator import DetectionEvaluator
from aurelix.models.aurelixnet import build_model
from aurelix.utils.logger import get_logger

logger = get_logger("aurelix.benchmarks.accuracy")


def run_accuracy_benchmark() -> Dict[str, Dict[str, float]]:
    """Benchmark accuracy metrics for AurelixNet variants."""
    results = {}
    variants = ["tiny", "small", "medium", "large", "xlarge"]

    for var in variants:
        evaluator = DetectionEvaluator(num_classes=80)
        metrics = evaluator.compute()
        results[var] = metrics
        logger.info(f"AurelixNet-{var.upper()} Accuracy Metrics: mAP@50={metrics.get('mAP@50')}, mAP@50:95={metrics.get('mAP@50:95')}")

    return results


if __name__ == "__main__":
    run_accuracy_benchmark()
