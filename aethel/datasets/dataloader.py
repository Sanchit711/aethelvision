"""Production Non-Blocking DataLoader Subsystem for Aethel Vision.

Implements high-performance, non-blocking PyTorch DataLoader construction with safe Windows
multiprocessing fallback, persistent workers, pin memory, prefetch factor, and seed worker initialization.
"""

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:
    torch = None  # type: ignore
    DataLoader = None  # type: ignore

from aethel.datasets.collator import BatchCollator, detection_collate_fn
from aethel.datasets.detection import DetectionDataset
from aethel.datasets.factory import DatasetFactory
from aethel.utils.logger import get_logger

logger = get_logger("aethel.datasets.dataloader")


def seed_worker(worker_id: int) -> None:
    """Worker initialization function ensuring reproducible random seeds across dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32 if torch is not None else 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset_path: Union[str, Path],
    batch_size: int = 16,
    num_workers: Optional[int] = None,
    split: str = "train",
    is_train: bool = True,
    shuffle: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
    config: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """Construct production non-blocking DataLoader with automated Windows fallback.

    Args:
        dataset_path: Path to dataset root directory.
        batch_size: Number of samples per batch.
        num_workers: Worker process count. Auto-calculated if None.
        split: Dataset split ('train', 'val', 'test').
        is_train: Flag for training vs evaluation mode.
        shuffle: Enable shuffling (defaults to True for train, False for val/test).
        pin_memory: Pin memory to host page-locked memory for fast CUDA transfers.
        persistent_workers: Maintain worker processes across epoch boundaries.
        prefetch_factor: Number of batches loaded in advance by each worker.
        config: Optional configuration settings.

    Returns:
        Instantiated PyTorch DataLoader.
    """
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is not installed or unavailable.")

    config = config or {}
    dataset = DatasetFactory.create(root=dataset_path, split=split, is_train=is_train, config=config)

    # 1. Determine optimal worker count
    if num_workers is None:
        cpu_cnt = os.cpu_count() or 1
        num_workers = min(4, cpu_cnt)

    # 2. Determine pin_memory & shuffling
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if shuffle is None:
        shuffle = is_train

    # 3. Determine persistent_workers & prefetch_factor
    if num_workers > 0:
        if persistent_workers is None:
            persistent_workers = True
        if prefetch_factor is None:
            prefetch_factor = 2
    else:
        persistent_workers = False
        prefetch_factor = None

    # 4. Windows safety check for multiprocessing
    is_windows = sys.platform == "win32"
    
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": min(batch_size, len(dataset)) if len(dataset) > 0 else batch_size,
        "shuffle": shuffle,
        "collate_fn": detection_collate_fn,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }

    try:
        if num_workers > 0:
            loader_kwargs["num_workers"] = num_workers
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor

        loader = DataLoader(**loader_kwargs)
        logger.info(f"Created DataLoader [{split}]: {len(dataset)} samples, batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}")
        return loader
    except Exception as e:
        if is_windows or num_workers > 0:
            logger.warning(f"Failed to spawn multi-worker DataLoader ({e}). Falling back safely to single-process (num_workers=0)...")
            loader_kwargs["num_workers"] = 0
            loader_kwargs["persistent_workers"] = False
            loader_kwargs.pop("prefetch_factor", None)
            loader_kwargs["worker_init_fn"] = None
            loader = DataLoader(**loader_kwargs)
            return loader
        raise e


def build_train_loader(dataset_path: Union[str, Path], batch_size: int = 16, num_workers: Optional[int] = None, config: Optional[Dict[str, Any]] = None) -> DataLoader:
    """Build training set DataLoader."""
    return create_dataloader(dataset_path=dataset_path, batch_size=batch_size, num_workers=num_workers, split="train", is_train=True, config=config)


def build_validation_loader(dataset_path: Union[str, Path], batch_size: int = 16, num_workers: Optional[int] = None, config: Optional[Dict[str, Any]] = None) -> DataLoader:
    """Build validation set DataLoader."""
    return create_dataloader(dataset_path=dataset_path, batch_size=batch_size, num_workers=num_workers, split="val", is_train=False, shuffle=False, config=config)


def build_test_loader(dataset_path: Union[str, Path], batch_size: int = 16, num_workers: Optional[int] = None, config: Optional[Dict[str, Any]] = None) -> DataLoader:
    """Build test set DataLoader."""
    return create_dataloader(dataset_path=dataset_path, batch_size=batch_size, num_workers=num_workers, split="test", is_train=False, shuffle=False, config=config)


__all__ = [
    "create_dataloader",
    "build_train_loader",
    "build_validation_loader",
    "build_test_loader",
    "seed_worker",
]
