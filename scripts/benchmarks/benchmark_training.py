"""Benchmark Training Speed and Iteration Time across AurelixNet Model Family.

Measures epoch throughput and batch step times across Tiny, Small, Medium, Large, and XLarge variants.
"""

import time
from typing import Dict

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None  # type: ignore

from aurelix.models.aurelixnet import build_model
from aurelix.trainer.trainer import Trainer
from aurelix.utils.logger import get_logger

logger = get_logger("aurelix.benchmarks.training")


def run_training_benchmark() -> Dict[str, float]:
    """Benchmark training step speed for AurelixNet variants."""
    if torch is None:
        return {}

    results = {}
    variants = ["tiny", "small", "medium", "large", "xlarge"]

    for var in variants:
        model = build_model(f"aurelixnet_{var}", num_classes=80)
        x = torch.randn(4, 3, 320, 320)
        y = {
            "cls_targets": torch.zeros(4, 80),
            "target_boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]).repeat(4, 1),
            "target_dfl": torch.tensor([[5.0, 5.0, 5.0, 5.0]]).repeat(4, 1),
        }

        # Measure 5 forward-backward steps
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        from aurelix.losses.composite import CompositeDetectionLoss

        loss_fn = CompositeDetectionLoss()

        t0 = time.perf_counter()
        for _ in range(5):
            optimizer.zero_grad()
            preds = model(x)
            loss, _ = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
        t1 = time.perf_counter()

        step_time_ms = ((t1 - t0) / 5.0) * 1000.0
        results[var] = round(step_time_ms, 2)
        logger.info(f"AurelixNet-{var.upper()} Training Step Time: {step_time_ms:.2f} ms / batch")

    return results


if __name__ == "__main__":
    run_training_benchmark()
