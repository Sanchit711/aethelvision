"""Generic Framework-Independent Registry Subsystem for Aethel Vision.

Allows plug-and-play registration of datasets, models, backbones, necks, heads,
losses, metrics, exporters, inference engines, optimizers, schedulers, and callbacks.
"""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from aethel.utils.exceptions import RegistryException
from aethel.utils.logger import get_logger

logger = get_logger("aethel.registry")

T = TypeVar("T")


class Registry:
    """Generic registry container for mapping string names to classes or functions."""

    def __init__(self, name: str) -> None:
        """Initialize named registry.

        Args:
            name: Human-readable name of the registry (e.g. 'MODEL_REGISTRY').
        """
        self._name = name
        self._module_dict: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Return the registry name."""
        return self._name

    def register(
        self,
        name: Optional[str] = None,
        force: bool = False,
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class or function into the registry.

        Args:
            name: Optional explicit key name. Defaults to class/function __name__.
            force: If True, overwrite existing registration key without throwing error.

        Returns:
            Decorator function.
        """

        def decorator(cls_or_fn: Type[T]) -> Type[T]:
            key = name if name is not None else cls_or_fn.__name__
            if key in self._module_dict and not force:
                raise RegistryException(
                    message=f"Module '{key}' is already registered in '{self._name}'.",
                    details=f"Existing: {self._module_dict[key]}, New: {cls_or_fn}",
                )
            self._module_dict[key] = cls_or_fn
            logger.debug(f"Registered '{key}' into '{self._name}'")
            return cls_or_fn

        return decorator

    def get(self, name: str) -> Any:
        """Retrieve a registered module by key name.

        Args:
            name: Registered key string.

        Returns:
            Registered class or function.
        """
        if name not in self._module_dict:
            raise RegistryException(
                message=f"Key '{name}' not found in registry '{self._name}'.",
                details=f"Available keys: {list(self._module_dict.keys())}",
            )
        return self._module_dict[name]

    def contains(self, name: str) -> bool:
        """Check if key exists in registry."""
        return name in self._module_dict

    def list_keys(self) -> List[str]:
        """List all registered key names."""
        return sorted(list(self._module_dict.keys()))

    def clear(self) -> None:
        """Clear all registered items (useful for isolation in testing)."""
        self._module_dict.clear()

    def __repr__(self) -> str:
        return f"<Registry '{self._name}' (Registered Count: {len(self._module_dict)})>"


# Pre-instantiated core registries
DATASET_REGISTRY = Registry("DATASET_REGISTRY")
MODEL_REGISTRY = Registry("MODEL_REGISTRY")
BACKBONE_REGISTRY = Registry("BACKBONE_REGISTRY")
NECK_REGISTRY = Registry("NECK_REGISTRY")
HEAD_REGISTRY = Registry("HEAD_REGISTRY")
LAYER_REGISTRY = Registry("LAYER_REGISTRY")
LOSS_REGISTRY = Registry("LOSS_REGISTRY")
TRAINER_REGISTRY = Registry("TRAINER_REGISTRY")
METRIC_REGISTRY = Registry("METRIC_REGISTRY")
PREDICTOR_REGISTRY = Registry("PREDICTOR_REGISTRY")
EXPORTER_REGISTRY = Registry("EXPORTER_REGISTRY")
OPTIMIZER_REGISTRY = Registry("OPTIMIZER_REGISTRY")
SCHEDULER_REGISTRY = Registry("SCHEDULER_REGISTRY")
CALLBACK_REGISTRY = Registry("CALLBACK_REGISTRY")
VISUALIZER_REGISTRY = Registry("VISUALIZER_REGISTRY")


__all__ = [
    "Registry",
    "DATASET_REGISTRY",
    "MODEL_REGISTRY",
    "BACKBONE_REGISTRY",
    "NECK_REGISTRY",
    "HEAD_REGISTRY",
    "LAYER_REGISTRY",
    "LOSS_REGISTRY",
    "TRAINER_REGISTRY",
    "METRIC_REGISTRY",
    "PREDICTOR_REGISTRY",
    "EXPORTER_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "CALLBACK_REGISTRY",
    "VISUALIZER_REGISTRY",
]
