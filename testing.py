import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.dataset import DoubleFrameKeyPointDataset, Rescale, RandomSquareCrop


ds_path = "/Users/milly/Documents/Personal/Projects/ML-Pose-Generation/dataset/Penn_Action"
dataset = DoubleFrameKeyPointDataset(
    ds_path, transforms=[Rescale(512), RandomSquareCrop(512)]
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

fig, axes = plt.subplots(2, 2)
for i, sample in enumerate(dataloader):
    print(type(sample["img1"]))
    axes[0, 0].imshow(np.moveaxis(np.squeeze(sample["img1"].numpy()), 0, 2))
    axes[0, 1].imshow(np.sum(np.squeeze(sample["labels1"].numpy()), axis=0))
    axes[1, 0].imshow(np.moveaxis(np.squeeze(sample["img2"].numpy()), 0, 2))
    axes[1, 1].imshow(np.sum(np.squeeze(sample["labels2"].numpy()), axis=0))

    plt.show()
    if plt.waitforbuttonpress():
        axes[0, 0].clear()
        axes[0, 1].clear()
        axes[1, 0].clear()
        axes[1, 1].clear()
