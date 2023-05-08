import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Callable, Tuple
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from .keypoints_transform import key_points_image


class PennActionABC(Dataset, ABC):

    """
    Abstract class for loading the PennAction dataset. Converts a single index
    to select a frame from multiple clips and the associated labels. Subclasses
    have to implement the `__getitem__` method. The methods `get_frame` and
    `get_labels` can be called after the method `get_frame_data` method to load
    the image as a PIL.Image object and the labels as a numpy array.
    """

    def __init__(
        self, root_dir: os.PathLike, transforms: Optional[List[Callable]] = None
    ):
        """
        Parameters
        ----------
        root_dir: path like
            Path to the dataset location.
        transforms: list, optional
            List of functions to apply to the images and the labels. Can be used
            for augementation or processing.

        """
        self.transforms = transforms

        self.root_dir = Path(root_dir)
        # PennAction dataset is split into directories of clips containing
        # a number of frames as images.
        # list of each clip directory
        clip_dirs = [
            dir_
            for dir_ in os.listdir(self.root_dir / "frames")
            if os.path.isdir(self.root_dir / "frames" / dir_)
        ]
        # save number of clips
        self.clip_n = len(clip_dirs)
        # The number of frames is saved in the label information under the
        # name "nframes"
        # getting the number of frames for each clip
        self.frames = np.array(
            [
                loadmat(path)["nframes"].item()
                for matfile in sorted(os.listdir(self.root_dir / "labels"))
                if (path := Path(self.root_dir / "labels" / matfile)).suffix
                == ".mat"
            ]
        )
        # this will be used to calculate which frame corresponds to each index
        self.cumulative_frames = np.zeros(len(self.frames) + 1, dtype=int)
        self.cumulative_frames[1:] = np.cumsum(self.frames)

        # This is populated when self.get_frame_data is called
        self.current_frame_info = {"clip": None, "frame": None}

    def __len__(self):
        return self.cumulative_frames[-1]

    # This has to be implemented in subclasses
    @abstractmethod
    def __getitem__(self, index):
        ...

    def get_frame_data(self, index: int) -> Tuple[int, int]:
        """
        Returns the clip number and frame number for a given `index`. They are
        also saved as instance attributes in the dictionary `current_frame_info`
        and can be accessed using the keys "clip" and "frame" respectively.

        Parameters
        ----------
        index: int

        Returns
        -------
        clip, frame: int, int
            The clip and frame numbers
        """

        # get the clip that the frame belongs to
        clip = np.digitize(index, self.cumulative_frames)
        self.current_frame_info["clip"] = clip

        # get the frame number within the clip
        frame = index - self.cumulative_frames[clip - 1] + 1
        self.current_frame_info["frame"] = frame

        return clip, frame

    def get_labels(
        self, clip: int, frame: int
    ) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """
        Returns the labels for a given clip and frame.

        Parameters
        ----------
        clip: int
            The clip number.
        frame: int
            The frame number

        Returns
        -------
        x, y: numpy.ndarray, numpy.ndarray
            The x and y coordinates of the landmark labels
        """

        label_path = self.root_dir / "labels" / f"{clip:04d}.mat"
        labels = loadmat(str(label_path))
        x = labels["x"][frame - 1]
        y = labels["y"][frame - 1]
        return x, y

    def get_frame(self, clip: int, frame: int) -> Image:
        """
        Returns the labels for a given frame.

        Parameters
        ----------
        clip: int
            The clip number.
        frame: int
            The frame number

        Returns
        -------
        img: PIL.Image
            The selected frame.
        """
        clip_dir = self.root_dir / "frames" / f"{clip:04d}"
        frame_path = clip_dir / f"{frame:06d}.jpg"
        img = Image.open(frame_path)
        return img


class SpatialKeyPointsPA(PennActionABC):
    """
    Converts the key point labels to 2D gaussian distributions with a seperate
    channel for each key point.
    """

    def __getitem__(self, index):
        clip, frame = self.get_frame_data(index)
        img = self.get_frame(clip, frame)
        x, y = self.get_labels(clip, frame)

        # apply transfomrms
        if self.transforms is not None:
            for transform in self.transforms:
                img, x, y = transform(img, x, y)

        # create gaussian images
        img = np.array(img)
        keypoint_img = key_points_image(img.shape[:2], x, y)

        # change axis order to match pytorch expected order
        # convert labels and image to torch,tensor.
        img = np.rollaxis(img, 2)
        keypoint_img = np.rollaxis(keypoint_img, 2)
        keypoint_img = keypoint_img.astype(np.float32)
        img = img / 255
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        keypoint_img = torch.from_numpy(keypoint_img)

        return {"img": img, "labels": keypoint_img}


class CoordKeyPointsPA(PennActionABC):

    """
    Keeps the key point labels in coordinate format but 0 centers them and
    scales them by the y axis.
    """

    def __getitem__(self, index):
        clip, frame = self.get_frame_data(index)
        img = self.get_frame(clip, frame)
        x, y = self.get_labels(clip, frame)

        # apply transforms
        if self.transforms is not None:
            for transform in self.transforms:
                img, x, y = transform(img, x, y)

        # Convert image to expected format
        img = np.array(img)
        img = np.rollaxis(img, 2)
        img = img / 255
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

        # zero center coordinates and scale by y axis size.
        size = img.shape[1]
        y = (y / size - 0.5).astype(np.float32)
        x = (x / size - (img.shape[0] / size) / 2).astype(np.float32)

        labels = np.stack([x, y], axis=1)
        labels = torch.from_numpy(labels)

        return {"img": img, "labels": labels}


# !!! EXPERIMENTAL !!!
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
