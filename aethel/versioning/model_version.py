"""Model Metadata and Versioning Subsystem for Aethel Vision Framework.

Provides ModelMetadataSpec embedded into checkpoints to guarantee complete traceability of model version,
tasks, class lists, dataset version, training configuration, export version, and framework version.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetadataSpec:
    """Complete metadata specification embedded into Aethel Vision model checkpoints."""

    model_version: str = "1.0.0"
    model_name: str = "AethelNet"
    tasks: List[str] = field(default_factory=lambda: ["detection"])
    class_list: List[str] = field(default_factory=list)
    dataset_version: str = "1.0"
    training_config: Dict[str, Any] = field(default_factory=dict)
    export_version: str = "1.0.0"
    framework_version: str = "1.1.0"
    checkpoint_epoch: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ModelMetadataSpec to dictionary."""
        return {
            "model_version": self.model_version,
            "model_name": self.model_name,
            "tasks": self.tasks,
            "class_list": self.class_list,
            "dataset_version": self.dataset_version,
            "training_config": self.training_config,
            "export_version": self.export_version,
            "framework_version": self.framework_version,
            "checkpoint_epoch": self.checkpoint_epoch,
            "metrics": self.metrics,
            "extra_metadata": self.extra_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadataSpec":
        """Deserialize ModelMetadataSpec from dictionary."""
        return cls(
            model_version=data.get("model_version", "1.0.0"),
            model_name=data.get("model_name", "AethelNet"),
            tasks=data.get("tasks", ["detection"]),
            class_list=data.get("class_list", []),
            dataset_version=data.get("dataset_version", "1.0"),
            training_config=data.get("training_config", {}),
            export_version=data.get("export_version", "1.0.0"),
            framework_version=data.get("framework_version", "1.1.0"),
            checkpoint_epoch=data.get("checkpoint_epoch", 0),
            metrics=data.get("metrics", {}),
            extra_metadata=data.get("extra_metadata", {}),
        )


__all__ = ["ModelMetadataSpec"]
