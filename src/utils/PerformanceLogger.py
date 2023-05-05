import os
from pathlib import Path
import logging
from typing import Optional
import pandas as pd


class PerformanceLogger:
    def __init__(self, filename: os.PathLike, name: Optional[str] = None):
        self.filename = Path(filename)
        if name is not None:
            name = f"Performance: {name}"
        else:
            name = "Performance"
        self.logger = logging.getLogger(name)

    def log_performance(
        self, epoch: int, mode: int, loss: float, accuracy: float
    ):
        message = (
            f"--- Epoch {epoch}: {mode}\n"
            f"------     Loss: {loss}\n"
            f"------ Accuracy: {accuracy}\n"
        )
        self.logger.log(logging.INFO, message)


class PandasPerformanceLogger(PerformanceLogger):
    def __init__(self, filename: os.PathLike, name: Optional[str] = None):
        super().__init__(filename, name)
        if self.filename.suffix != ".csv":
            self.df_filename = self.df_filename.with_suffix(".csv")

        self.data = {"epoch": [], "mode": [], "loss": [], "accuracy": []}

    def log_performance(
        self, epoch: int, mode: int, loss: float, accuracy: float
    ):
        super().log_performance(epoch, mode, loss, accuracy)
        self.data["epoch"].append(epoch)
        self.data["mode"].append(mode)
        self.data["loss"].append(loss)
        self.data["accuracy"].append(accuracy)

        df = pd.DataFrame(self.data)
        df.to_csv(self.df_filename)
