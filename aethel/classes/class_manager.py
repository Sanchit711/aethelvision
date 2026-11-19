"""Class Management Subsystem for Aethel Vision Framework.

Provides ClassSpec, ClassAlias, and ClassManager to support dynamic class addition,
removal, renaming, alias mapping, and class metadata tracking without breaking previous checkpoints.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ClassSpec:
    """Class specification record."""

    class_id: int
    name: str
    aliases: Set[str] = field(default_factory=set)
    supercategory: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ClassSpec to dictionary."""
        return {
            "class_id": self.class_id,
            "name": self.name,
            "aliases": sorted(list(self.aliases)),
            "supercategory": self.supercategory,
            "metadata": self.metadata,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassSpec":
        """Deserialize ClassSpec from dictionary."""
        return cls(
            class_id=data["class_id"],
            name=data["name"],
            aliases=set(data.get("aliases", [])),
            supercategory=data.get("supercategory"),
            metadata=data.get("metadata", {}),
            active=data.get("active", True),
        )


class ClassManager:
    """Manages class taxonomies, aliases, additions, removals, and backward compatibility mappings."""

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        """Initialize ClassManager.

        Args:
            class_names: Initial list of class names.
        """
        self._classes: Dict[int, ClassSpec] = {}
        self._name_to_id: Dict[str, int] = {}
        self._alias_to_id: Dict[str, int] = {}

        if class_names:
            for idx, name in enumerate(class_names):
                self.add_class(name=name, class_id=idx)

    @property
    def num_classes(self) -> int:
        """Return total active class count."""
        return len([c for c in self._classes.values() if c.active])

    @property
    def total_capacity(self) -> int:
        """Return total allocated class slots (including removed/inactive)."""
        return len(self._classes)

    def add_class(
        self,
        name: str,
        class_id: Optional[int] = None,
        aliases: Optional[List[str]] = None,
        supercategory: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClassSpec:
        """Add a new class to the taxonomy.

        Args:
            name: Class name string.
            class_id: Optional explicit class ID. If None, auto-assign next slot.
            aliases: Optional list of alias strings for this class.
            supercategory: Optional parent category name.
            metadata: Optional class metadata dictionary.

        Returns:
            Created ClassSpec object.
        """
        if name in self._name_to_id:
            existing_id = self._name_to_id[name]
            spec = self._classes[existing_id]
            spec.active = True
            if aliases:
                spec.aliases.update(aliases)
                for a in aliases:
                    self._alias_to_id[a.lower()] = existing_id
            return spec

        cid = class_id if class_id is not None else len(self._classes)
        alias_set = set(aliases) if aliases else set()

        spec = ClassSpec(
            class_id=cid,
            name=name,
            aliases=alias_set,
            supercategory=supercategory,
            metadata=metadata or {},
            active=True,
        )

        self._classes[cid] = spec
        self._name_to_id[name] = cid
        self._alias_to_id[name.lower()] = cid
        for a in alias_set:
            self._alias_to_id[a.lower()] = cid

        return spec

    def remove_class(self, name_or_id: Any) -> bool:
        """Deactivate a class from active inference/training without renumbering existing IDs."""
        cid = self.get_class_id(name_or_id)
        if cid is not None and cid in self._classes:
            self._classes[cid].active = False
            return True
        return False

    def rename_class(self, old_name: str, new_name: str) -> bool:
        """Rename an existing class while maintaining backward compatibility alias to old name."""
        cid = self.get_class_id(old_name)
        if cid is None or cid not in self._classes:
            return False

        spec = self._classes[cid]
        spec.aliases.add(old_name)
        self._alias_to_id[old_name.lower()] = cid

        if old_name in self._name_to_id:
            del self._name_to_id[old_name]

        spec.name = new_name
        self._name_to_id[new_name] = cid
        self._alias_to_id[new_name.lower()] = cid
        return True

    def get_class_id(self, query: Any) -> Optional[int]:
        """Resolve class ID by integer ID, class name, or alias."""
        if isinstance(query, int):
            return query if query in self._classes else None
        if isinstance(query, str):
            if query in self._name_to_id:
                return self._name_to_id[query]
            if query.lower() in self._alias_to_id:
                return self._alias_to_id[query.lower()]
        return None

    def get_class_name(self, class_id: int) -> str:
        """Get canonical name for class ID."""
        if class_id in self._classes:
            return self._classes[class_id].name
        return f"class_{class_id}"

    def get_active_class_names(self) -> List[str]:
        """Get list of active class names sorted by class_id."""
        sorted_ids = sorted(self._classes.keys())
        return [self._classes[i].name for i in sorted_ids if self._classes[i].active]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ClassManager taxonomy state."""
        return {
            "classes": {str(cid): spec.to_dict() for cid, spec in self._classes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassManager":
        """Deserialize ClassManager taxonomy state."""
        mgr = cls()
        raw_classes = data.get("classes", {})
        for cid_str, spec_dict in raw_classes.items():
            spec = ClassSpec.from_dict(spec_dict)
            mgr._classes[spec.class_id] = spec
            mgr._name_to_id[spec.name] = spec.class_id
            mgr._alias_to_id[spec.name.lower()] = spec.class_id
            for a in spec.aliases:
                mgr._alias_to_id[a.lower()] = spec.class_id
        return mgr


__all__ = ["ClassSpec", "ClassManager"]
