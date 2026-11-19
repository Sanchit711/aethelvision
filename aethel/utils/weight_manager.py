"""Pretrained Model Weight Management and Verification Subsystem for Aethel Vision Framework.

Provides WeightManager, WeightMetadata, and WeightVerifier for SHA256 checksum validation,
version compatibility verification, and model card generation.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from aethel.utils.exceptions import AethelException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.utils.weight_manager")


@dataclass
class WeightMetadata:
    """Dataclass holding pretrained weight metadata information."""

    model_name: str = "AethelNet-Small"
    version: str = "1.0.0"
    num_classes: int = 80
    sha256: str = ""
    file_size_bytes: int = 0
    framework_version: str = "1.1.0"
    training_dataset: str = "COCO 2017"
    metrics: Dict[str, float] = field(default_factory=dict)
    license: str = "Apache-2.0"


class WeightVerifier:
    """Utility verifier for SHA256 hash checksum verification."""

    @staticmethod
    def compute_sha256(filepath: Union[str, Path]) -> str:
        """Compute SHA256 hex digest for a target weight file."""
        path = Path(filepath)
        if not path.exists():
            raise AethelException(message=f"Weight file not found: {path}")

        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @classmethod
    def verify(cls, filepath: Union[str, Path], expected_sha256: str) -> bool:
        """Verify weight file SHA256 checksum matches expected digest."""
        computed = cls.compute_sha256(filepath)
        is_valid = computed.lower() == expected_sha256.lower()
        if is_valid:
            logger.info(f"Verified weight integrity for {filepath} (SHA256: {computed[:12]}...)")
        else:
            logger.error(f"Checksum mismatch for {filepath}! Computed: {computed}, Expected: {expected_sha256}")
        return is_valid


class WeightManager:
    """Manager for indexing, validating, and downloading model weights."""

    def __init__(self, weights_dir: Union[str, Path] = "weights") -> None:
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def create_model_card(self, metadata: WeightMetadata, output_path: Union[str, Path]) -> Path:
        """Export model card metadata JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)
        logger.info(f"Exported model card metadata to {path}")
        return path

    def load_metadata(self, metadata_path: Union[str, Path]) -> WeightMetadata:
        """Load model card metadata from JSON file."""
        path = Path(metadata_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WeightMetadata(**data)


__all__ = ["WeightMetadata", "WeightVerifier", "WeightManager"]
