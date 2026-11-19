"""DataLoader Builders for Mhyra Vision Object Detection Platform.

Provides high-performance PyTorch DataLoader builder functions:
- build_train_loader()
- build_validation_loader()
- build_test_loader()
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:
    torch = None  # type: ignore
    DataLoader = None  # type: ignore

from datasets.collate import collate_fn
from datasets.factory import DatasetFactory
from trainer.configuration import TrainingConfig


def build_train_loader(
    data_path_or_config: Union[str, Path, TrainingConfig],
    dataset_type: str = "auto",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    img_size: Optional[Tuple[int, int]] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = True,
    shuffle: bool = True,
    **kwargs: Any,
) -> Any:
    """Build and return training PyTorch DataLoader instance."""
    if isinstance(data_path_or_config, TrainingConfig):
        cfg = data_path_or_config
        data_path = cfg.dataset_path
        batch_size = batch_size or cfg.batch_size
        num_workers = num_workers if num_workers is not None else cfg.num_workers
        pin_memory = cfg.pin_memory
        persistent_workers = cfg.persistent_workers
        prefetch_factor = cfg.prefetch_factor
        if isinstance(cfg.img_size, tuple):
            img_size = img_size or cfg.img_size
        else:
            img_size = img_size or (cfg.img_size, cfg.img_size)
    else:
        data_path = data_path_or_config
        batch_size = batch_size or 16
        num_workers = num_workers if num_workers is not None else 4
        img_size = img_size or (640, 640)

    dataset = DatasetFactory.create_dataset(
        data_path_or_manifest=data_path,
        dataset_type=dataset_type,
        split="train",
        img_size=img_size,
        is_train=True,
        **kwargs,
    )

    if DataLoader is None:
        return dataset

    loader_kwargs: dict = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


def build_validation_loader(
    data_path_or_config: Union[str, Path, TrainingConfig],
    dataset_type: str = "auto",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    img_size: Optional[Tuple[int, int]] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    shuffle: bool = False,
    **kwargs: Any,
) -> Any:
    """Build and return validation PyTorch DataLoader instance."""
    if isinstance(data_path_or_config, TrainingConfig):
        cfg = data_path_or_config
        data_path = cfg.dataset_path
        batch_size = batch_size or cfg.batch_size
        num_workers = num_workers if num_workers is not None else cfg.num_workers
        pin_memory = cfg.pin_memory
        persistent_workers = cfg.persistent_workers
        prefetch_factor = cfg.prefetch_factor
        if isinstance(cfg.img_size, tuple):
            img_size = img_size or cfg.img_size
        else:
            img_size = img_size or (cfg.img_size, cfg.img_size)
    else:
        data_path = data_path_or_config
        batch_size = batch_size or 16
        num_workers = num_workers if num_workers is not None else 4
        img_size = img_size or (640, 640)

    dataset = DatasetFactory.create_dataset(
        data_path_or_manifest=data_path,
        dataset_type=dataset_type,
        split="val",
        img_size=img_size,
        is_train=False,
        **kwargs,
    )

    if DataLoader is None:
        return dataset

    loader_kwargs: dict = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


def build_test_loader(
    data_path_or_config: Union[str, Path, TrainingConfig],
    dataset_type: str = "auto",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    img_size: Optional[Tuple[int, int]] = None,
    pin_memory: bool = True,
    **kwargs: Any,
) -> Any:
    """Build and return testing PyTorch DataLoader instance."""
    return build_validation_loader(
        data_path_or_config=data_path_or_config,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
        **kwargs,
    )


__all__ = ["build_train_loader", "build_validation_loader", "build_test_loader"]
