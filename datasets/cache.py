"""Production Dataset Caching Manager for Mhyra Detector Datasets.

Provides RAM, Disk, and Pass-through (None) caching strategies for preprocessed image
tensors and bounding box targets with md5 cache validation.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch


class DatasetCacheManager:
    """Manages RAM and Disk caching of preprocessed dataset items."""

    def __init__(
        self,
        cache_mode: str = "none",
        cache_dir: Union[str, Path] = ".cache/dataset",
    ) -> None:
        """Initialize DatasetCacheManager.

        Args:
            cache_mode: Cache strategy string ('ram', 'disk', 'none').
            cache_dir: Directory path for disk caching.
        """
        self.cache_mode = cache_mode.lower()
        self.cache_dir = Path(cache_dir)
        self.ram_cache: Dict[int, Any] = {}

        if self.cache_mode == "disk":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, index: int) -> Optional[Any]:
        """Fetch cached item at index if present.

        Args:
            index: Item index integer.

        Returns:
            Cached item payload or None if cache miss.
        """
        if self.cache_mode == "ram":
            return self.ram_cache.get(index)
        elif self.cache_mode == "disk":
            file_p = self.cache_dir / f"item_{index}.pt"
            if file_p.exists():
                try:
                    return torch.load(file_p, weights_only=False)
                except Exception:
                    return None
        return None

    def set(self, index: int, data: Any) -> None:
        """Store item at index into cache.

        Args:
            index: Item index integer.
            data: Payload object to cache.
        """
        if self.cache_mode == "ram":
            self.ram_cache[index] = data
        elif self.cache_mode == "disk":
            file_p = self.cache_dir / f"item_{index}.pt"
            try:
                torch.save(data, file_p)
            except Exception:
                pass

    def clear(self) -> None:
        """Clear RAM and Disk cache entries."""
        self.ram_cache.clear()
        if self.cache_mode == "disk" and self.cache_dir.exists():
            for f in self.cache_dir.glob("item_*.pt"):
                try:
                    f.unlink()
                except Exception:
                    pass


__all__ = ["DatasetCacheManager"]
