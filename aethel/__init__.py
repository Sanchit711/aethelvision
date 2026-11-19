"""Aethel Vision Framework.

Enterprise Computer Vision Framework for Edge AI & Cloud Intelligence.
"""

__version__ = "1.0.0"
__author__ = "Aethel Vision Engineering Team"
__license__ = "Apache-2.0"

from aethel.utils.logger import get_logger, setup_logger
from aethel.utils.exceptions import (
    AethelException,
    DeviceMismatchException,
    TensorDeviceMismatchException,
    EMADeviceMismatchException,
)
from aethel.facade import Aethel, AethelResultWrapper

__all__ = [
    "Aethel",
    "AethelResultWrapper",
    "__version__",
    "get_logger",
    "setup_logger",
    "AethelException",
    "DeviceMismatchException",
    "TensorDeviceMismatchException",
    "EMADeviceMismatchException",
]
