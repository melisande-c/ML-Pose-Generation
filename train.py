import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from models.DoubleUNet import DoubleUNet
from data.dataset import FrameKeyPointDataset, Rescale, RandomSquareCrop

# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
epochs = 3
ds_path = "/Users/milly/Documents/Personal/Projects/ML-Pose-Generation/dataset/Penn_Action"

model = DoubleUNet()
model.to(device)

dataset = FrameKeyPointDataset(
    ds_path, transforms=[Rescale(512), RandomSquareCrop(512)]
)
dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
optimiser = Adam(model.parameters(), lr=0.001)
loss1 = nn.L1Loss()
loss2 = nn.L1Loss()

for epoch in range(epochs):
    print(f"--- Epoch: {epoch}")

    total_l1 = 0
    total_l2 = 0
    dn = 0
    for i, sample in enumerate(dataloader):
        for name in sample:
            sample[name] = sample[name].to(device)

        model.train()
        model.zero_grad()

        labels1, img2 = model(sample["img1"], sample["labels2"])

        l1 = loss1(labels1, sample["labels1"])
        l2 = loss2(img2, sample["img2"])
        l = l1 + l2
        l.backward()
        optimiser.step()

        dn += 1

        print(f"--- Batch Loss 1: {l1:.4f}")
        print(f"--- Batch Loss 2: {l2:.4f}")
    print(f"--- Epoch Loss 1: {total_l1/dn:.4f}")
    print(f"--- Epoch Loss 2: {total_l2/dn:.4f}")
