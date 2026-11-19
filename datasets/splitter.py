"""Production Dataset Splitter Utility for Mhyra Detector Datasets.

Splits detection datasets into reproducible Train, Validation, and Test subsets
using configurable random seeds and split ratios.
"""

import random
from pathlib import Path
from typing import List, Tuple, Union


class DatasetSplitter:
    """Utility for partitioning dataset file pairs into train, val, and test splits."""

    @staticmethod
    def split_paths(
        img_paths: List[Union[str, Path]],
        label_paths: List[Union[str, Path]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[
        Tuple[List[Path], List[Path]],
        Tuple[List[Path], List[Path]],
        Tuple[List[Path], List[Path]],
    ]:
        """Split dataset file paths into train, val, and test sets.

        Args:
            img_paths: List of image file paths.
            label_paths: List of corresponding label file paths.
            train_ratio: Proportion of dataset for training (default 0.8).
            val_ratio: Proportion of dataset for validation (default 0.1).
            test_ratio: Proportion of dataset for testing (default 0.1).
            seed: Random seed for shuffling (default 42).

        Returns:
            Tuple of ((train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls)).
        """
        assert len(img_paths) == len(label_paths), "Mismatch between img_paths and label_paths length"

        pairs = list(zip([Path(p) for p in img_paths], [Path(p) for p in label_paths]))

        rng = random.Random(seed)
        rng.shuffle(pairs)

        n_total = len(pairs)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train : n_train + n_val]
        test_pairs = pairs[n_train + n_val :]

        def unzip(pair_list):
            if not pair_list:
                return [], []
            imgs, lbls = zip(*pair_list)
            return list(imgs), list(lbls)

        return unzip(train_pairs), unzip(val_pairs), unzip(test_pairs)


__all__ = ["DatasetSplitter"]
