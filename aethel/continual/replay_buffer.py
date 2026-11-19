"""Replay Buffer Subsystem for Continual Learning in Aethel Vision Framework.

Provides ReplayBuffer to retain historical exemplars across dataset versions and fine-tuning runs,
preventing catastrophic forgetting of previously learned classes.
"""

import random
from typing import Any, Dict, List, Optional, Tuple


class ReplayBuffer:
    """Experience replay buffer storing historical dataset exemplars."""

    def __init__(self, max_size: int = 1000, strategy: str = "balanced") -> None:
        """Initialize ReplayBuffer.

        Args:
            max_size: Maximum total capacity of exemplar items.
            strategy: Selection/eviction strategy ('balanced', 'fifo', 'random').
        """
        self.max_size = max_size
        self.strategy = strategy
        self._buffer: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, sample: Dict[str, Any]) -> None:
        """Add an exemplar sample record into buffer.

        Args:
            sample: Sample dictionary containing image data, labels, bboxes, and metadata.
        """
        if len(self._buffer) >= self.max_size:
            if self.strategy == "fifo":
                self._buffer.pop(0)
            elif self.strategy == "random":
                idx = random.randint(0, len(self._buffer) - 1)
                self._buffer.pop(idx)
            else:
                # Balanced eviction by default
                self._buffer.pop(0)

        self._buffer.append(sample)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of historical exemplars.

        Args:
            batch_size: Number of items to sample.

        Returns:
            List of sampled exemplar records.
        """
        if not self._buffer:
            return []
        k = min(batch_size, len(self._buffer))
        return random.sample(self._buffer, k)

    def clear(self) -> None:
        """Clear all buffer items."""
        self._buffer.clear()


__all__ = ["ReplayBuffer"]
