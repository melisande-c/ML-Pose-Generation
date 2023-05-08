from PIL import Image
import numpy as np
from numpy.typing import NDArray
from torchvision import transforms


class Rescale:

    """
    Rescales an image (and coordinates) so that the smallest dimension is
    equal to the instance attribute size.
    """

    def __init__(self, size: int):
        """
        Parameters
        ----------
        size: int
            The size that the smallest dimension of the image will be rescaled
            to.
        """
        self.size = size

    def __call__(self, img: Image, x: NDArray, y: NDArray):
        """
        Parameters
        ----------
        img: PIL.Image
        x: numpy.ndarray
            x coordinates of landmarks.
        y: numpy.ndarray
            y coordinates of landmarks.
        """
        axis = np.argmin(img.size)
        ratio = self.size / img.size[axis]

        new_size = [int(img.size[1] * ratio), int(img.size[0] * ratio)]
        new_size[abs(axis - 1)] = self.size
        new_size = tuple(new_size)
        img = transforms.Resize(new_size)(img)

        # make sure to scale key points
        x = x * ratio
        y = y * ratio
        return img, x, y


class RandomSquareCrop:

    """
    Crops the image to a random square.
    """

    def __init__(self, size: int):
        """
        Paramters
        ---------
        size: int
            The size to crop the square to.
        """
        self.size = size

    def __call__(self, img, x, y):
        """
        Parameters
        ----------
        img: PIL.Image
        x: numpy.ndarray
            x coordinates of landmarks.
        y: numpy.ndarray
            y coordinates of landmarks.
        """

        # the buffer space between the desired square size and the image dims
        idiff = max(0, img.size[1] - self.size - 1)
        jdiff = max(0, img.size[0] - self.size - 1)

        # choose the starting point of the square in both dimensions
        if idiff != 0:
            i = np.random.randint(0, idiff)
        else:
            i = 0
        if jdiff != 0:
            j = np.random.randint(0, jdiff)
        else:
            j = 0

        # crop the image
        img = np.array(img)[i : i + self.size, j : j + self.size]
        img = Image.fromarray(img)

        # calculate the new keypoint coords by subtracting the offset.
        x = x - j
        y = y - i

        return img, x, y
