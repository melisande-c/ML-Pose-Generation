import os
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.io import loadmat
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from .keypoints_transform import key_points_image


class FrameKeyPointDataset(Dataset):
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

    def __getitem__(self, index):
        clip = np.digitize(index, self.cumulative_frames)
        self.current_frame_info["clip"] = clip

        clip_dir = self.root_dir / "frames" / f"{clip:04d}"

        label_path = self.root_dir / "labels" / f"{clip:04d}.mat"
        labels = loadmat(str(label_path))

        frame = index - self.cumulative_frames[clip - 1] + 1
        self.current_frame_info["frame"] = frame

        frame_data = self._get_frame(clip_dir, labels, frame)

        return frame_data

    def _get_frame(self, clip_dir, labels, frame):
        frame_path = clip_dir / f"{frame:06d}.jpg"
        img = Image.open(frame_path)

        x = labels["x"][frame - 1]
        y = labels["y"][frame - 1]

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


class DoubleFrameKeyPointDataset(FrameKeyPointDataset):
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


class Rescale:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img, x, y):
        axis = np.argmin(img.size)
        ratio = self.size / img.size[axis]

        new_size = (int(img.size[1] * ratio), int(img.size[0] * ratio))
        img = transforms.Resize(new_size)(img)

        x = x * ratio
        y = y * ratio
        return img, x, y


class RandomSquareCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img, x, y):
        idiff = max(0, img.size[1] - self.size)
        jdiff = max(0, img.size[0] - self.size)

        if idiff != 0:
            i = np.random.randint(0, idiff)
        else:
            i = 0
        if jdiff != 0:
            j = np.random.randint(0, jdiff)
        else:
            j = 0

        img = np.array(img)[i : i + self.size, j : j + self.size]
        img = Image.fromarray(img)

        x = x - j
        y = y - i

        return img, x, y
