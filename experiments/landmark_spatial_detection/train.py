import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from models.VGGMod import VGGMod
from data.dataset import FrameKeyPointDataset, Rescale, RandomSquareCrop


# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
epochs = 3
ds_path = "/Users/milly/Documents/Personal/Projects/ML-Pose-Generation/dataset/Penn_Action"

vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg19", pretrained=True)
model = VGGMod(vgg, 512, 13)
model.to(device)

dataset = FrameKeyPointDataset(
    ds_path, transforms=[Rescale(512), RandomSquareCrop(512)]
)
dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
optimiser = Adam(model.parameters(), lr=0.001)
loss = nn.L1Loss()
