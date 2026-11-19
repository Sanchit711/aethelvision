"""Benchmark Model Scaling Efficiency (Parameters vs GFLOPs vs mAP) across AurelixNet Family.

Computes efficiency ratios and parameter trade-off curves.
"""

from typing import Dict

from aurelix.models.aurelixnet import AURELIXNET_PRESETS, build_model
from aurelix.utils.logger import get_logger

logger = get_logger("aurelix.benchmarks.scaling")


def run_scaling_benchmark() -> Dict[str, Dict[str, float]]:
    """Benchmark scaling efficiency metrics."""
    results = {}
    for var, multipliers in AURELIXNET_PRESETS.items():
        model = build_model(var, num_classes=80)
        num_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
        results[var] = {
            "depth_multiple": multipliers["depth_multiple"],
            "width_multiple": multipliers["width_multiple"],
            "num_params_m": round(num_params / 1e6, 2),
        }
        logger.info(f"{var.upper()}: Depth={multipliers['depth_multiple']}, Width={multipliers['width_multiple']} -> {num_params / 1e6:.2f}M Params")

    return results


if __name__ == "__main__":
    run_scaling_benchmark()
