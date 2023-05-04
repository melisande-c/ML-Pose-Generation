import os
import pathlib
from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Optimizer
from typing import TYPE_CHECKING, Optional
from .PerformanceLogger import PerformanceLogger

if TYPE_CHECKING:
    Loss = nn.modules.loss._Loss


class ModelTrainer(ABC):
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
        self.model = model
        self.dataset = dataset
        self.dataset_train, self.dataset_test = random_split(
            dataset, [train_fraction, 1 - train_fraction]
        )
        self.dataloader_train = DataLoader(
            self.dataloader_train, batch_size, shuffle=True
        )
        self.dataloader_test = DataLoader(
            self.dataloader_test, batch_size, shuffle=True
        )
        self.optimiser = optimiser
        self.loss = loss
        self.savedir = pathlib.Path(savedir)
        if logger is not None:
            self.logger = PerformanceLogger()
        else:
            self.logger = logger

        self.epoch = 0
        self.train_loss = None
        self.test_loss = None

    @abstractmethod
    def output(self, sample):
        ...

    @abstractmethod
    def calc_loss(self, output, sample) -> torch.Tensor:
        self

    def train(self):
        self.model.train()
        self.model.zero_grad()
        running_loss = 0
        for i, sample in enumerate(self.dataloader_train):
            output = self.output(sample)
            loss = self.calc_loss(output, sample)
            running_loss += loss
            loss.backward()
            self.optimiser.step()
        self.train_loss = running_loss / i

    def test(self):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0
            for i, sample in enumerate(self.dataloader_train):
                output = self.output(sample)
                loss = self.calc_loss(output, sample)
                running_loss += loss
            self.test_loss = running_loss / i

    def run(self, epochs: int):
        for epoch in range(epochs):
            self.epoch += 1
            self.train()
            self.logger.log_performance(
                epoch, "train", self.train_loss.item(), None
            )
            self.test()
            self.logger.log_performance(
                epoch, "test", self.test_loss.item(), None
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimiser.state_dict(),
                    "loss": self.train_loss,
                },
                self.savedir / "model",
            )
