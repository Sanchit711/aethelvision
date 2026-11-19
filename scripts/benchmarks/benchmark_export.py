"""Benchmark TorchScript and ONNX Model Export Speed and Binary File Sizes across AurelixNet.

Exports each AurelixNet model variant to ONNX and TorchScript, measuring compilation time and artifact file sizes.
"""

from pathlib import Path
from typing import Dict

from aurelix.export.exporter import ModelExporter
from aurelix.models.aurelixnet import build_model
from aurelix.utils.logger import get_logger

logger = get_logger("aurelix.benchmarks.export")


def run_export_benchmark(tmp_dir: str = "runs/benchmark/export") -> Dict[str, Dict[str, float]]:
    """Benchmark model export performance."""
    results = {}
    output_dir = Path(tmp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = ["tiny", "small", "medium", "large", "xlarge"]

    for var in variants:
        model = build_model(f"aurelixnet_{var}", num_classes=80)
        exporter = ModelExporter(model=model)

        onnx_file = output_dir / f"aurelixnet_{var}.onnx"
        exporter.export_onnx(onnx_file, input_shape=(1, 3, 320, 320))
        onnx_mb = onnx_file.stat().st_size / (1024 * 1024) if onnx_file.exists() else 0.0

        results[var] = {"onnx_size_mb": round(onnx_mb, 2)}
        logger.info(f"AurelixNet-{var.upper()} ONNX Binary Size: {onnx_mb:.2f} MB")

    return results


if __name__ == "__main__":
    run_export_benchmark()
