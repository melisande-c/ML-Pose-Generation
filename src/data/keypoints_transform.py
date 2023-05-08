from typing import Tuple, Optional
from numpy.typing import NDArray
import numpy as np


def make_gaussian(
    shape: Tuple[int, int],
    fwhm: float = 16,
    center: Optional[Tuple[int, int]] = None,
) -> NDArray[np.float_]:
    """
    Creates a 2D gausian.

    Parameters
    ----------
    shape: (int, int)
        Output array shape
    fwhm: float
        The full width at half maximum of the distribution
    center: (int, int), optional
        The center of the distribution, if None, it will default to the center
        of the array

    Returns
    -------
    gaussian_image: numpy.ndarray

    """
    i, j = np.mgrid[: shape[0], : shape[1]]

    if center is None:
        i0 = shape[0] // 2
        j0 = shape[1] // 2
    else:
        i0 = center[0]
        j0 = center[1]

    return np.exp(-4 * np.log(2) * ((i - i0) ** 2 + (j - j0) ** 2) / fwhm**2)


def key_points_image(
    shape: Tuple[int, int],
    key_points_x: NDArray,
    key_points_y: NDArray,
    fwhm: float = 16,
):
    """
    Creates a channel for each key point and creates a gaussian with the center
    at the key point

    Parameters
    ----------
    shape: (int, int)
        The 2D shape of the image.
    key_points_x: numpy.ndarray
        The x coordinates of the key points.
    key_points_y: numpy.ndarray
        The y coordinates of the key points.
    fwhm: float
        The full width at half maximum of the distrubutions.
    """
    output = np.zeros((*shape, key_points_x.shape[0]))
    for i, (x, y) in enumerate(zip(key_points_x, key_points_y)):
        output[..., i] = make_gaussian(shape, fwhm, (y, x))
    return output
