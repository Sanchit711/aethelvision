"""Automated Model Packaging and Release System for Aurelix Vision Framework.

Packages PyTorch checkpoint (.pt), ONNX binary (.onnx), TorchScript binary (.torchscript),
metadata card (metadata.json), and SHA256 checksums into a release artifact directory.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Union

from aurelix.export.exporter import ModelExporter
from aurelix.models.aurelixnet import build_model
from aurelix.utils.logger import get_logger
from aurelix.utils.weight_manager import WeightMetadata, WeightVerifier

logger = get_logger("aurelix.scripts.package_model")


def package_model(
    model_name: str = "aurelixnet_small",
    num_classes: int = 80,
    output_dir: Union[str, Path] = "releases/aurelixnet_small",
) -> Path:
    """Package model artifacts into release directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate model
    model = build_model(model_name, num_classes=num_classes)
    exporter = ModelExporter(model=model)

    # 2. Save PyTorch weights
    pt_path = out_path / "model.pt"
    model.save_weights(pt_path)

    # 3. Export ONNX and TorchScript binaries
    onnx_path = out_path / "model.onnx"
    exporter.export_onnx(onnx_path, input_shape=(1, 3, 320, 320))

    ts_path = out_path / "model.torchscript"
    exporter.export_torchscript(ts_path, input_shape=(1, 3, 320, 320))

    # 4. Generate SHA256 checksums
    pt_hash = WeightVerifier.compute_sha256(pt_path)
    onnx_hash = WeightVerifier.compute_sha256(onnx_path)
    ts_hash = WeightVerifier.compute_sha256(ts_path)

    checksums = {
        "model.pt": pt_hash,
        "model.onnx": onnx_hash,
        "model.torchscript": ts_hash,
    }

    with open(out_path / "checksums.json", "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    # 5. Export metadata card
    metadata = WeightMetadata(
        model_name=model_name,
        num_classes=num_classes,
        sha256=pt_hash,
        file_size_bytes=pt_path.stat().st_size,
    )

    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata.__dict__, f, indent=2)

    # 6. Generate README.md card
    readme_content = f"""# {model_name.upper()} Release Package

## Artifact Integrity Checksums
- `model.pt`: `{pt_hash}`
- `model.onnx`: `{onnx_hash}`
- `model.torchscript`: `{ts_hash}`

## Usage Instructions
Load PyTorch model checkpoint programmatically using Aurelix Vision Framework.
"""
    with open(out_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"Successfully packaged release artifacts to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="aurelixnet_small")
    parser.add_argument("--classes", type=int, default=80)
    parser.add_argument("--output", type=str, default="releases/aurelixnet_small")
    args = parser.parse_args()

    package_model(model_name=args.model, num_classes=args.classes, output_dir=args.output)
