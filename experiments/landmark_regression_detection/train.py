import sys
import os
from pathlib import Path
import argparse
import logging
from typing import Literal

sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
from torch import nn
from torch.optim import Adam
from src.models.VGGMod import VGGCoordMod
from src.data.datasets import CoordKeyPointsPA
from src.data.transforms import Rescale, RandomSquareCrop
from src.utils.ModelTrainer import ModelTrainer
from src.utils.PerformanceLogger import PandasPerformanceLogger


class VGGTrainer(ModelTrainer):

    """
    VGG trainer subclass of model trainer
    """

    # define how input sample to model
    def output(self, sample):
        return self.model(sample["img"].to(self.device))

    # define how to calculate loss
    def calc_loss(self, output, sample) -> torch.Tensor:
        return self.loss(output, sample["labels"].to(self.device))


class EuclideanDistanceLoss(nn.Module):

    """
    Calculates the mean euclidean distance between the predicted keypoints and
    the target keypoints.
    """

    def __init__(self, reduction=Literal["sum", "mean", "none"]) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, output, target):
        pdist = nn.PairwiseDistance(p=2)
        dist = pdist(output, target)
        if self.reduction == "mean":
            return torch.mean(dist)
        elif self.reduction == "sum":
            return torch.sum(dist)
        elif self.reduction == "none":
            return dist
        else:
            raise ValueError(
                f"Reduction value, '{self.reduction}' not accepted."
            )


# info logger
logger = logging.getLogger(
    logging.basicConfig(
        format="%(asctime)-10s  %(name)-40s  %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        force=True,
    )
)


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
        "-e",
        "--epochs",
        help="Number of Epochs to train for.",
        default=5,
        type=int,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="No. of inputs per batch.",
        default=32,
        type=int,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate of optimiser.",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "-tf",
        "--training_fraction",
        help="Fraction of dataset to use for training.",
        default=0.75,
        type=float,
    )
    args = parser.parse_args()

    # make output dir
    args.save_dir = Path(args.save_dir)
    for parent in args.save_dir.parents[::-1]:
        if not os.path.isdir(parent):
            os.mkdir(parent)
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    logger.info("Getting model")
    vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=True)
    logger.info("Customising model")
    model = VGGCoordMod(vgg, 13)

    logger.info("Initialising objects")
    dataset = CoordKeyPointsPA(
        # rescale and crop data to VGG expected input (224, 224)
        args.dataset_path,
        transforms=[Rescale(224), RandomSquareCrop(224)],
    )
    optimiser = Adam(model.parameters(), lr=args.learning_rate)
    loss = EuclideanDistanceLoss(reduction="sum")

    performance_logger = PandasPerformanceLogger(
        args.save_dir / "performance.csv"
    )

    logger.info("Beginning training")
    trainer = VGGTrainer(
        model,
        dataset,
        optimiser,
        loss,
        args.save_dir,
        args.batch_size,
        train_fraction=args.training_fraction,
        logger=performance_logger,
    )

    # train model
    trainer.run(args.epochs)


if __name__ == "__main__":
    main()
