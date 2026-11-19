"""Aethel Security & Cryptographic Watermarking Submodule."""

from aethel.security.watermark import (
    AethelWatermarkEngine,
    watermark_model,
    verify_model_provenance,
)

__all__ = [
    "AethelWatermarkEngine",
    "watermark_model",
    "verify_model_provenance",
]
