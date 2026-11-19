"""Aethel Detector: Proprietary Object Detection Architecture Package.

Provides the complete AethelDetector neural network model interface, backbone,
neck, head, loss functions, post-processing decoders, and export pipelines.
"""

from aethel_detector.model import AethelDetector

__version__ = "0.1.0"
__all__ = ["AethelDetector"]
