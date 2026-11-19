"""Dataset Versioning Subsystem for Aethel Vision Framework.

Provides DatasetVersionSpec to track dataset iterations (V1, V2, V3), added classes,
removed classes, updated labels, annotation counts, and dataset quality statistics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetVersionSpec:
    """Dataset version specification metadata."""

    version_id: str  # e.g., "v1.0", "v2.1"
    dataset_name: str
    num_samples: int = 0
    num_annotations: int = 0
    added_classes: List[str] = field(default_factory=list)
    removed_classes: List[str] = field(default_factory=list)
    updated_labels_count: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    quality_score: float = 1.0
    parent_version: Optional[str] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DatasetVersionSpec to dictionary."""
        return {
            "version_id": self.version_id,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "num_annotations": self.num_annotations,
            "added_classes": self.added_classes,
            "removed_classes": self.removed_classes,
            "updated_labels_count": self.updated_labels_count,
            "class_distribution": self.class_distribution,
            "quality_score": self.quality_score,
            "parent_version": self.parent_version,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetVersionSpec":
        """Deserialize DatasetVersionSpec from dictionary."""
        return cls(
            version_id=data["version_id"],
            dataset_name=data.get("dataset_name", "default_dataset"),
            num_samples=data.get("num_samples", 0),
            num_annotations=data.get("num_annotations", 0),
            added_classes=data.get("added_classes", []),
            removed_classes=data.get("removed_classes", []),
            updated_labels_count=data.get("updated_labels_count", 0),
            class_distribution=data.get("class_distribution", {}),
            quality_score=data.get("quality_score", 1.0),
            parent_version=data.get("parent_version"),
            checksum=data.get("checksum"),
            metadata=data.get("metadata", {}),
        )


__all__ = ["DatasetVersionSpec"]
