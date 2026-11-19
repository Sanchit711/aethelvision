"""Aethel Datasets Submodule.

Universal dataset subsystem for computer vision pipelines.
"""

from aethel.datasets.base import BaseDataset
from aethel.datasets.cache import DatasetCache, DiskDatasetCache
from aethel.datasets.collator import BatchCollator
from aethel.datasets.dataloader import (
    build_test_loader,
    build_train_loader,
    build_validation_loader,
    create_dataloader,
)
from aethel.datasets.detection import DetectionDataset
from aethel.datasets.factory import DatasetFactory
from aethel.datasets.loader import AnnotationLoader, ImageLoader
from aethel.datasets.metadata import AnnotationInfo, DatasetMetadata, ImageInfo
from aethel.datasets.parser import DatasetParser, LabelParser
from aethel.datasets.statistics import DatasetStatistics
from aethel.datasets.transforms import TransformPipeline
from aethel.datasets.validator import DatasetIntegrityChecker, DatasetValidator
from aethel.datasets.visualization import VisualizationManager

__all__ = [
    "BaseDataset",
    "DetectionDataset",
    "DatasetFactory",
    "create_dataloader",
    "build_train_loader",
    "build_validation_loader",
    "build_test_loader",
    "DatasetParser",
    "LabelParser",
    "DatasetStatistics",
    "DatasetValidator",
    "DatasetIntegrityChecker",
    "ImageLoader",
    "AnnotationLoader",
    "BatchCollator",
    "VisualizationManager",
    "TransformPipeline",
    "DatasetCache",
    "DiskDatasetCache",
    "DatasetMetadata",
    "ImageInfo",
    "AnnotationInfo",
]

