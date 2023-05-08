from torch.utils.data import Subset
import numpy as np
from .datasets import PennActionABC


def train_test_split(dataset: PennActionABC, train_fraction):
    n = len(dataset.cumulative_frames) - 1
    train_n = int(round(train_fraction * n))
    train_clip_idx = np.random.choice(n, train_n, replace=False)
    train_idx = np.array(tuple(), dtype=int)
    test_idx = np.array(tuple(), dtype=int)
    for i in range(n):
        if i in train_clip_idx:
            train_idx = np.concatenate(
                [
                    train_idx,
                    np.arange(
                        dataset.cumulative_frames[i],
                        dataset.cumulative_frames[i + 1],
                    ),
                ],
            )
        else:
            test_idx = np.concatenate(
                [
                    test_idx,
                    np.arange(
                        dataset.cumulative_frames[i],
                        dataset.cumulative_frames[i + 1],
                    ),
                ],
            )
    train_ds = Subset(dataset, train_idx)
    test_ds = Subset(dataset, test_idx)
    return train_ds, test_ds
