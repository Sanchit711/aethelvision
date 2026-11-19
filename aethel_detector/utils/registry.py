"""Component Registry for Aethel Detector Architecture.

Responsibility:
Registers backbone, neck, head, and loss modules for dynamic model instantiation.
"""

from typing import Any, Callable, Dict


class ComponentRegistry:
    """Registry mapping component type names to class constructors."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Any], Any]:
        """Decorator registering a class or function by string name."""
        def decorator(cls_or_func: Any) -> Any:
            self._registry[name] = cls_or_func
            return cls_or_func
        return decorator

    def get(self, name: str) -> Any:
        """Retrieve registered constructor by name."""
        if name not in self._registry:
            raise KeyError(f"Component '{name}' not found in registry '{self.name}'.")
        return self._registry[name]


BACKBONES = ComponentRegistry("backbones")
NECKS = ComponentRegistry("necks")
HEADS = ComponentRegistry("heads")
LOSSES = ComponentRegistry("losses")
