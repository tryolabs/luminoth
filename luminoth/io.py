import numpy as np
import os

from PIL import Image


def read_image(path):
    """Reads an image located at `path` into an array.

    Arguments:
        path (str): Path to a valid image file in the filesystem.

    Returns:
        `numpy.ndarray` of size `(height, width, channels)`.
    """
    full_path = os.path.expanduser(path)
    return np.array(Image.open(full_path).convert('RGB'))
