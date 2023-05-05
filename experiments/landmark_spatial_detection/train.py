import sys
import os
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
from torch import nn
from torch.optim import Adam
from src.models.VGGMod import VGGMod
from src.data.dataset import FrameKeyPointDataset, Rescale, RandomSquareCrop
from src.utils.ModelTrainer import ModelTrainer
from src.utils.PerformanceLogger import PandasPerformanceLogger


class VGGTrainer(ModelTrainer):
    # define how input sample to model
    def output(self, sample):
        return self.model(sample["img"].to(self.device))

    # define how to calculate loss
    def calc_loss(self, output, sample) -> torch.Tensor:
        return self.loss(output, sample["labels"].to(self.device))


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

    args.save_dir = Path(args.save_dir)
    for parent in args.save_dir.parents[::-1]:
        if not os.path.isdir(parent):
            os.mkdir(parent)
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    logger.info("Getting model")
    vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=True)
    logger.info("Customising model")
    model = VGGMod(vgg, 512, 13)

    logger.info("Initialising objects")
    dataset = FrameKeyPointDataset(
        args.dataset_path, transforms=[Rescale(224), RandomSquareCrop(224)]
    )
    optimiser = Adam(model.parameters(), lr=args.learning_rate)
    loss = nn.L1Loss()

    performance_logger = PandasPerformanceLogger(
        args.save_dir / "performance.csv"
    )

    logger.info("Beggining training")
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

    trainer.run(args.epochs)


if __name__ == "__main__":
    main()
