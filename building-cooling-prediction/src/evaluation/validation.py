from typing import Iterator, Tuple, Optional

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split


def simple_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def k_fold_split(
    X,
    y,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        yield train_idx, test_idx


def time_series_split(
    X,
    y,
    n_splits: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Time-series aware train/test indices.

    Ensures that the training indices are always before the validation indices
    to prevent data leakage when dealing with temporal data.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        yield train_idx, test_idx
