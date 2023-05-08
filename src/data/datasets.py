import os
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.io import loadmat
import torch

from torch.utils.data import Dataset
from .keypoints_transform import key_points_image


class PennActionABC(Dataset, ABC):
    def __init__(self, root_dir: os.PathLike, transforms=None):
        self.transforms = transforms

        self.root_dir = Path(root_dir)
        clip_dirs = [
            dir_
            for dir_ in os.listdir(self.root_dir / "frames")
            if os.path.isdir(self.root_dir / "frames" / dir_)
        ]
        self.clip_n = len(clip_dirs)
        self.frames = np.array(
            [
                loadmat(path)["nframes"].item()
                for matfile in sorted(os.listdir(self.root_dir / "labels"))
                if (path := Path(self.root_dir / "labels" / matfile)).suffix
                == ".mat"
            ]
        )
        self.cumulative_frames = np.zeros(len(self.frames) + 1, dtype=int)
        self.cumulative_frames[1:] = np.cumsum(self.frames)

        self.current_frame_info = {"clip": None, "frame": None}

    def __len__(self):
        return self.cumulative_frames[-1]

    @abstractmethod
    def __getitem__(self, index):
        ...

    def get_frame_data(self, index: int):
        clip = np.digitize(index, self.cumulative_frames)
        self.current_frame_info["clip"] = clip

        frame = index - self.cumulative_frames[clip - 1] + 1
        self.current_frame_info["frame"] = frame

        return clip, frame

    def get_labels(self, clip: int, frame: int):
        label_path = self.root_dir / "labels" / f"{clip:04d}.mat"
        labels = loadmat(str(label_path))
        x = labels["x"][frame - 1]
        y = labels["y"][frame - 1]
        return x, y

    def get_frame(self, clip: int, frame: int):
        clip_dir = self.root_dir / "frames" / f"{clip:04d}"
        frame_path = clip_dir / f"{frame:06d}.jpg"
        img = Image.open(frame_path)
        return img


class SpatialKeyPointsPA(PennActionABC):
    def __getitem__(self, index):
        clip, frame = self.get_frame_data(index)
        img = self.get_frame(clip, frame)
        x, y = self.get_labels(clip, frame)

        if self.transforms is not None:
            for transform in self.transforms:
                img, x, y = transform(img, x, y)

        img = np.array(img)
        keypoint_img = key_points_image(img.shape[:2], x, y)

        img = np.rollaxis(img, 2)
        keypoint_img = np.rollaxis(keypoint_img, 2)
        keypoint_img = keypoint_img.astype(np.float32)
        img = img / 255
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        keypoint_img = torch.from_numpy(keypoint_img)

        return {"img": img, "labels": keypoint_img}


class CoordKeyPointsPA(PennActionABC):
    def __getitem__(self, index):
        clip, frame = self.get_frame_data(index)
        img = self.get_frame(clip, frame)
        x, y = self.get_labels(clip, frame)

        if self.transforms is not None:
            for transform in self.transforms:
                img, x, y = transform(img, x, y)

        img = np.array(img)
        img = np.rollaxis(img, 2)
        img = img / 255
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

        size = img.shape[1]
        y = (y / size - 0.5).astype(np.float32)
        x = (x / size - (img.shape[0] / size) / 2).astype(np.float32)

        labels = np.stack([x, y], axis=1)
        labels = torch.from_numpy(labels)

        return {"img": img, "labels": labels}


class DoubleSpatialKeypointsPA(SpatialKeyPointsPA):
    def __getitem__(self, index):
        frame_data_1 = super().__getitem__(index)

        clip = self.current_frame_info["clip"]
        frame1 = self.current_frame_info["frame"]
        frame_n = (
            self.cumulative_frames[clip] - self.cumulative_frames[clip - 1]
        )
        mid = frame_n // 2
        frame2 = (frame1 + mid) % frame_n
        index2 = self.cumulative_frames[clip - 1] + frame2
        frame_data_2 = super().__getitem__(index2)

        return {
            "img1": frame_data_1["img"],
            "img2": frame_data_2["img"],
            "labels1": frame_data_1["labels"],
            "labels2": frame_data_2["labels"],
        }
