from __future__ import annotations
import os
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Any
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from .PerformanceLogger import PerformanceLogger
from ..data.train_test_split import train_test_split

if TYPE_CHECKING:
    Loss = nn.modules.loss._Loss

info_logger = logging.getLogger(__name__)


class ModelTrainer(ABC):

    """
    Abstract class for training models and logging progress. The methods,
    `output` and `calc_loss` have to be implemented by subclasses.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        optimiser: Optimizer,
        loss: Loss,
        savedir: os.PathLike,
        batch_size: int = 64,
        train_fraction: float = 0.75,
        logger: Optional[PerformanceLogger] = None,
    ):
        """
        Parameters
        ----------
        model: nn.Module
            The model to train.
        dataset: Dataset
            The torch dataset to train and test on.
        optimiser: Optimizer
            The optimiser used for optimising the model.
        loss: Loss
            The critereon for the model to improve by.
        save_dir: path like
            The directory to save the trained models and the training metadata
            such as the loss and accuracy to.
        batch_size: int
            The batch size.
        train_fraction: float
            The fraction of the dataset to use for training, the remainder will
            be used for testing.
        logger: PerformanceLogger, optional
            The logger used for saving & displaying performance during training.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        info_logger.info(f"Using device: {self.device}")
        self.model = model
        self.model.to(self.device)
        self.dataset = dataset
        self.dataset_train, self.dataset_test = train_test_split(
            dataset, train_fraction
        )
        self.dataloader_train = DataLoader(
            self.dataset_train, batch_size, shuffle=True
        )
        self.dataloader_test = DataLoader(
            self.dataset_test, batch_size, shuffle=True
        )
        self.optimiser = optimiser
        self.loss = loss
        self.savedir = pathlib.Path(savedir)
        if logger is None:
            self.logger = PerformanceLogger(self.savedir / "Performance.log")
        else:
            self.logger = logger

        self.epoch = 0
        self.train_loss = None
        self.test_loss = None

    @abstractmethod
    def output(self, sample: Any):
        """
        Function to get the output of the model from a given sample of the
        dataset.

        Parameters
        ----------
        sample: Any
            The output when iterating through the data loader.
        """

    @abstractmethod
    def calc_loss(self, output: torch.tensor, sample: Any) -> torch.Tensor:
        """
        Function for calculating the loss from the output of the model from
        a given sample

        Parameters
        ----------
        output: torch.tensor
            The output of the model from the `sample`.
        sample: any
            A sample from the dataset.
        """

    def train(self):
        """
        A function to train one epoch.
        """

        self.model.train()  # make sure model is in training mode
        self.model.zero_grad()  # reset gradients
        # keep the running loss for the whole epoch
        running_loss = 0
        n = 0
        for i, sample in enumerate(self.dataloader_train):
            output = self.output(sample)  # get the output from the model
            loss = self.calc_loss(output, sample)  # calculate the loss
            loss.backward()  # gradient back propagation
            self.optimiser.step()  # optimise
            # detach so the current graph can be garbage collected
            running_loss += loss.detach()
        self.train_loss = running_loss / len(
            self.dataset_train
        )  # divide by i for the mean

    def test(self):
        self.model.eval()  # evaluation mode
        # don't caclulate gradients during testing.
        with torch.no_grad():
            running_loss = 0
            for i, sample in enumerate(self.dataloader_train):
                output = self.output(sample)
                loss = self.calc_loss(output, sample)
                running_loss += loss.detach()
            self.test_loss = running_loss / len(self.dataset_test)

    def run(self, epochs: int):
        for epoch in range(epochs):
            self.epoch += 1
            info_logger.info(f"--- Epoch {self.epoch}")
            # train model
            self.train()
            self.logger.log_performance(
                self.epoch, "train", self.train_loss.item(), None
            )
            # test model
            self.test()
            self.logger.log_performance(
                self.epoch, "test", self.test_loss.item(), None
            )
            # save output at each epoch
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimiser.state_dict(),
                    "loss": self.train_loss,
                },
                self.savedir / "model",
            )
