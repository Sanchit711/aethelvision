"""Task Registry Subsystem for Aethel Vision Framework.

Provides TaskSpec data structures and TaskRegistry container to register multi-task model heads,
losses, dataset parsers, metrics, decoders, and visualizers dynamically.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from aethel.registry.registry import Registry
from aethel.utils.exceptions import RegistryException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.registry.task")


@dataclass
class TaskSpec:
    """Task specification registering all components required for a vision task."""

    task_name: str
    head_cls: Optional[Any] = None
    loss_cls: Optional[Any] = None
    dataset_parser_cls: Optional[Any] = None
    metric_evaluator_cls: Optional[Any] = None
    decoder_cls: Optional[Any] = None
    visualizer_cls: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TaskSpec metadata."""
        return {
            "task_name": self.task_name,
            "head_cls": self.head_cls.__name__ if hasattr(self.head_cls, "__name__") else str(self.head_cls),
            "loss_cls": self.loss_cls.__name__ if hasattr(self.loss_cls, "__name__") else str(self.loss_cls),
            "dataset_parser_cls": (
                self.dataset_parser_cls.__name__
                if hasattr(self.dataset_parser_cls, "__name__")
                else str(self.dataset_parser_cls)
            ),
            "metric_evaluator_cls": (
                self.metric_evaluator_cls.__name__
                if hasattr(self.metric_evaluator_cls, "__name__")
                else str(self.metric_evaluator_cls)
            ),
            "decoder_cls": self.decoder_cls.__name__ if hasattr(self.decoder_cls, "__name__") else str(self.decoder_cls),
            "visualizer_cls": (
                self.visualizer_cls.__name__
                if hasattr(self.visualizer_cls, "__name__")
                else str(self.visualizer_cls)
            ),
            "metadata": self.metadata,
        }


class TaskRegistryContainer(Registry):
    """Registry container specifically designed for multi-task specifications."""

    def __init__(self) -> None:
        super().__init__("TASK_REGISTRY")

    def register_task(self, task_spec: TaskSpec, force: bool = False) -> TaskSpec:
        """Register a TaskSpec instance.

        Args:
            task_spec: TaskSpec object containing task component mappings.
            force: If True, overwrite existing task spec.

        Returns:
            The registered TaskSpec instance.
        """
        key = task_spec.task_name.lower()
        if key in self._module_dict and not force:
            raise RegistryException(
                message=f"Task '{key}' is already registered in TASK_REGISTRY.",
                details=f"Existing: {self._module_dict[key]}, New: {task_spec}",
            )
        self._module_dict[key] = task_spec
        logger.info(f"Registered TaskSpec '{key}' in TASK_REGISTRY")
        return task_spec

    def get_task(self, task_name: str) -> TaskSpec:
        """Retrieve registered TaskSpec by name."""
        key = task_name.lower()
        if key not in self._module_dict:
            raise RegistryException(
                message=f"Task '{task_name}' is not registered in TASK_REGISTRY.",
                details=f"Available tasks: {self.list_keys()}",
            )
        return self._module_dict[key]


TASK_REGISTRY = TaskRegistryContainer()

__all__ = ["TaskSpec", "TaskRegistryContainer", "TASK_REGISTRY"]
