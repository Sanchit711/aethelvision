"""Head Subpackage for Aethel Detector."""

from aethel_detector.head.decoder import HeadDecoder
from aethel_detector.head.detect import DecoupledDetectHead

__all__ = ["DecoupledDetectHead", "HeadDecoder"]
