"""Checkpoint and Model Version Migration Subsystem for Aethel Vision Framework.

Provides ModelVersionMigrator to automatically upgrade legacy v0.3.x / v1.0 checkpoints
to v1.1+ multi-task, continually-learned checkpoint structures.
"""

from typing import Any, Dict, Optional, Tuple

from aethel.versioning.model_version import ModelMetadataSpec
from aethel.utils.logger import get_logger

logger = get_logger("aethel.continual.migration")


class ModelVersionMigrator:
    """Migrates legacy or previous-version model state dicts and metadata specs to current framework format."""

    @staticmethod
    def migrate_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate input checkpoint to current version format.

        Args:
            checkpoint: Raw state dictionary loaded from PyTorch checkpoint.

        Returns:
            Upgraded checkpoint dictionary containing ModelMetadataSpec metadata.
        """
        migrated = dict(checkpoint)

        # Detect or construct metadata spec
        raw_meta = migrated.get("metadata")
        if isinstance(raw_meta, dict):
            meta_spec = ModelMetadataSpec.from_dict(raw_meta)
        else:
            # Reconstruct metadata from legacy keys
            class_list = migrated.get("class_names", migrated.get("classes", []))
            num_classes = migrated.get("num_classes", len(class_list) if class_list else 80)
            if not class_list:
                class_list = [f"class_{i}" for i in range(num_classes)]

            meta_spec = ModelMetadataSpec(
                model_version=migrated.get("version", "1.0.0"),
                model_name=migrated.get("model_name", "AethelDetector"),
                tasks=["detection"],
                class_list=class_list,
                dataset_version=migrated.get("dataset_version", "1.0"),
                checkpoint_epoch=migrated.get("epoch", 0),
            )

        migrated["metadata"] = meta_spec.to_dict()
        logger.info(
            f"Migrated checkpoint (Model: {meta_spec.model_name}, Tasks: {meta_spec.tasks}, "
            f"Classes: {len(meta_spec.class_list)})"
        )
        return migrated


__all__ = ["ModelVersionMigrator"]
