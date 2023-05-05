from __future__ import annotations
import os
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Optimizer
from .PerformanceLogger import PerformanceLogger

if TYPE_CHECKING:
    Loss = nn.modules.loss._Loss

train_logger = logging.getLogger("TrainLogger")


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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        train_logger.info(f"Using device: {self.device}")
        self.model = model
        self.model.to(self.device)
        self.dataset = dataset
        self.dataset_train, self.dataset_test = random_split(
            dataset, [train_fraction, 1 - train_fraction]
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
            train_logger.info(f"--- Epoch {self.epoch}")
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
