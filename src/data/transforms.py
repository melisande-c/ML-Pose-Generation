from PIL import Image
import numpy as np
from torchvision import transforms


class Rescale:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img, x, y):
        axis = np.argmin(img.size)
        ratio = self.size / img.size[axis]

        new_size = [int(img.size[1] * ratio), int(img.size[0] * ratio)]
        new_size[abs(axis - 1)] = self.size
        new_size = tuple(new_size)
        img = transforms.Resize(new_size)(img)

        x = x * ratio
        y = y * ratio
        return img, x, y


class RandomSquareCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img, x, y):
        idiff = max(0, img.size[1] - self.size - 1)
        jdiff = max(0, img.size[0] - self.size - 1)

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
