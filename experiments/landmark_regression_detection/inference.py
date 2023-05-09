import sys
import os
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.models.VGGMod import VGGCoordMod
from src.data.datasets import CoordKeyPointsPA
from src.data.transforms import Rescale, RandomSquareCrop
from src.data.train_test_split import train_test_split


def convert_coords(coords, shape):
    size = shape[1]
    coords[:, 1] = (coords[:, 1] + 0.5) * size
    coords[:, 0] = (coords[:, 0] + (shape[0] / size) / 2) * shape[1]
    return coords


def create_figure(sample, output):
    img = sample["img"].cpu().numpy()
    img = np.squeeze(img)
    shape = img.shape
    img = np.moveaxis(img, 0, 2)

    coords = np.squeeze(sample["labels"].cpu().numpy())
    coords = convert_coords(coords, shape)
    output = np.squeeze(output.cpu().numpy())
    pred_coords = convert_coords(output, shape)

    fig, axes = plt.subplots(1, 2)
    for ax in axes.flatten():
        ax.imshow(img)
        ax.axis("off")

    axes[0].plot(coords[:, 0], coords[:, 1], "ro")
    axes[0].set_title("Labels")

    axes[1].plot(pred_coords[:, 0], pred_coords[:, 1], "bo")
    axes[1].set_title("Predicted")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dsp", "--dataset_path", help="Path to dataset.", required=True
    )
    parser.add_argument(
        "-mp", "--model_path", help="Path to model", required=True
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        help="Directory to save models and logs to.",
        required=True,
    )
    parser.add_argument(
        "-tf",
        "--training_fraction",
        help="Fraction of dataset to use for training.",
        default=0.75,
        type=float,
    )
    parser.add_argument("-n", "--number_of_samples", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # make output dir
    args.save_dir = Path(args.save_dir)
    for parent in args.save_dir.parents[::-1]:
        if not os.path.isdir(parent):
            os.mkdir(parent)
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    dataset = CoordKeyPointsPA(
        # rescale and crop data to VGG expected input (224, 224)
        args.dataset_path,
        transforms=[Rescale(224), RandomSquareCrop(224)],
    )
    dataset_train, dataset_test = train_test_split(
        dataset, args.training_fraction
    )
    del dataset_train
    dataloader = DataLoader(dataset_test, 1, shuffle=True)

    vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=False)
    model = VGGCoordMod(vgg, 13)

    model_data = torch.load(args.model_path, map_location="cpu")
    model_state_dict = model_data["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    for i, sample in enumerate(dataloader):
        if i > args.number_of_samples:
            break

        with torch.no_grad():
            for key, value in sample.items():
                sample[key] = value.to(device)

            output = model(sample["img"])
            fig = create_figure(sample, output)

            fig.savefig(Path(args.save_dir) / f"{i}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
