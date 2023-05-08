from typing import Tuple
import numpy as np
from torch.utils.data import Subset
from .datasets import PennActionABC

# for repeatablility
np.random.seed(42)


def train_test_split(
    dataset: PennActionABC, train_fraction: float
) -> Tuple[Subset, Subset]:
    """
    For splitting the PennAction data set into a train and test subset. Clips
    are not split between train and test since frames within a clip are not
    independent of each other and would give an inflated performance on the test
    set.

    Parameters
    ----------
    dataset: PennActionABC
        The dataset.
    train_fraction: float
        The fraction of the clips to be used for training. The remainder will be
        used for testing.
    """
    # number of clips
    n = len(dataset.cumulative_frames) - 1
    # number of training clips
    train_n = int(round(train_fraction * n))
    # randomly chosen training clips
    train_clip_idx = np.random.choice(n, train_n, replace=False)

    # iterate through the clip indices and calculate the frame indices from
    # the cumulative frames attribute.
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
    # return subsets.
    train_ds = Subset(dataset, train_idx)
    test_ds = Subset(dataset, test_idx)
    return train_ds, test_ds
