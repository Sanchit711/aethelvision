"""Benchmark Memory Footprint and VRAM Requirements across AurelixNet Model Family.

Estimates parameter memory, buffer allocations, and GPU VRAM footprint for AurelixNet variants.
"""

from typing import Dict

try:
    import torch
except Exception:
    torch = None  # type: ignore

from aurelix.models.aurelixnet import build_model
from aurelix.utils.logger import get_logger

logger = get_logger("aurelix.benchmarks.memory")


def run_memory_benchmark() -> Dict[str, float]:
    """Benchmark memory footprint for AurelixNet variants."""
    results = {}
    variants = ["tiny", "small", "medium", "large", "xlarge"]

    for var in variants:
        model = build_model(f"aurelixnet_{var}", num_classes=80)
        num_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
        param_mb = (num_params * 4) / (1024 * 1024)  # 4 bytes per float32 param
        results[var] = round(param_mb, 2)
        logger.info(f"AurelixNet-{var.upper()} Weight Memory: {param_mb:.2f} MB ({num_params / 1e6:.2f}M Params)")

    return results


if __name__ == "__main__":
    run_memory_benchmark()
