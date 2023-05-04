import sys
from pathlib import Path
import argparse

print(str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from src.models.VGGMod import VGGMod
from src.data.dataset import FrameKeyPointDataset, Rescale, RandomSquareCrop
from src.utils.ModelTrainer import ModelTrainer
from src.utils.PerformanceLogger import PandasPerformanceLogger


class VGGTrainer(ModelTrainer):
    # define how input sample to model
    def output(self, sample):
        return self.model(sample["img"])

    # define how to calculate loss
    def calc_loss(self, output, sample) -> torch.Tensor:
        return self.loss(output, sample["labels"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dsp", "--dataset_path", help="Path to dataset.", required=True
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        help="Directory to save models and logs to.",
        required=True,
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of Epochs to train for.", default=5
    )
    parser.add_argument(
        "-bs", "--batch_size", help="No. of inputs per batch.", default=32
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate of optimiser.",
        default=0.001,
    )
    parser.add_argument(
        "-tf",
        "--training_fraction",
        help="Fraction of dataset to use for training.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Getting model")
    vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg19", pretrained=True)
    model = VGGMod(vgg, 512, 13)
    model.to(device)

    print("Initialising objects")
    dataset = FrameKeyPointDataset(
        args.dataset_path, transforms=[Rescale(512), RandomSquareCrop(512)]
    )
    optimiser = Adam(model.parameters(), lr=args.learning_rate)
    loss = nn.L1Loss()

    logger = PandasPerformanceLogger(args.save_dir / "performance.csv")

    print("Beggining training")
    trainer = VGGTrainer(
        model,
        dataset,
        optimiser,
        loss,
        args.save_dir,
        args.batch_size,
        train_fraction=args.train_fraction,
        logger=logger,
    )

    trainer.run(args.epochs)


if __name__ == "__main__":
    main()
