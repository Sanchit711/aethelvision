"""Dataset Caching Subsystem for Aethel Vision.

Provides optional in-memory image buffer caching for high-speed training iterations.
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from aethel.datasets.metadata import ImageInfo
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.cache")


class DatasetCache:
    """In-memory image cache container."""

    def __init__(self, max_mb: float = 2048.0, enabled: bool = False) -> None:
        """Initialize DatasetCache.

        Args:
            max_mb: Maximum memory capacity in megabytes (default 2GB).
            enabled: Master toggle flag for caching.
        """
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.enabled = enabled
        self._cache: Dict[str, np.ndarray] = {}
        self._current_bytes = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """Fetch array from cache if present."""
        if not self.enabled:
            return None
        return self._cache.get(key)

    def put(self, key: str, image: np.ndarray) -> bool:
        """Store array in cache if space permits.

        Returns:
            True if cached successfully, False if memory capacity exceeded.
        """
        if not self.enabled:
            return False

        size_bytes = image.nbytes
        if self._current_bytes + size_bytes > self.max_bytes:
            logger.debug(f"Cache memory limit ({self.max_bytes / 1e6:.1f} MB) reached. Skipping cache for {key}")
            return False

        self._cache[key] = image
        self._current_bytes += size_bytes
        return True

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._current_bytes = 0

    def stats(self) -> Dict[str, Union[int, float]]:
        """Return cache memory usage statistics."""
        return {
            "cached_count": len(self._cache),
            "used_mb": self._current_bytes / (1024 * 1024),
            "max_mb": self.max_bytes / (1024 * 1024),
            "usage_percent": (self._current_bytes / self.max_bytes * 100.0) if self.max_bytes > 0 else 0.0,
        }


class DiskDatasetCache:
    """Persistent disk cache for dataset parsing and scanning acceleration (.cache)."""

    @staticmethod
    def compute_signature(root_dir: Path, split: str = "train") -> str:
        """Compute MD5 dataset signature based on folder path, count, and modification times."""
        root = Path(root_dir)
        hasher = hashlib.md5()
        hasher.update(str(root.resolve()).encode("utf-8"))
        hasher.update(split.encode("utf-8"))

        if root.exists():
            for p in sorted(root.glob("*")):
                try:
                    st = p.stat()
                    hasher.update(f"{p.name}:{st.st_mtime}:{st.st_size}".encode("utf-8"))
                except Exception:
                    pass
        return hasher.hexdigest()

    @staticmethod
    def load_samples(cache_file: Path, current_signature: str) -> Optional[List[ImageInfo]]:
        """Load parsed ImageInfo samples from disk cache file if signature matches."""
        cache_p = Path(cache_file)
        if not cache_p.exists():
            return None

        try:
            with open(cache_p, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and data.get("signature") == current_signature:
                logger.info(f"Successfully loaded dataset from persistent disk cache: {cache_p}")
                return data.get("samples")
            else:
                logger.info(f"Dataset signature changed. Rebuilding disk cache at {cache_p}...")
                return None
        except Exception as e:
            logger.warning(f"Could not load disk cache file {cache_p}: {e}. Re-scanning...")
            return None

    @staticmethod
    def save_samples(cache_file: Path, current_signature: str, samples: List[ImageInfo]) -> bool:
        """Save parsed ImageInfo samples to disk cache file atomically."""
        cache_p = Path(cache_file)
        try:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = cache_p.with_suffix(".tmp")
            payload = {"signature": current_signature, "samples": samples}
            with open(tmp_file, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_file.replace(cache_p)
            logger.info(f"Saved {len(samples)} samples to persistent disk cache at {cache_p}")
            return True
        except Exception as e:
            logger.warning(f"Failed to write persistent disk cache: {e}")
            return False


__all__ = ["DatasetCache", "DiskDatasetCache"]

