import numpy as np
from skimage import color

from albumentations.core.transforms_interface import ImageOnlyTransform

class RGB2Lab(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img, **params):
        img = color.rgb2lab(img)
        return img

class RGB2YCbCr(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, img: np.ndarray, **params):
        img = color.rgb2ycbcr(img)
        return img