import numpy as np
from skimage import color

from albumentations.core.transforms_interface import ImageOnlyTransform

class RGB2Lab(ImageOnlyTransform):
    """Convert ndarray image to ndarray Lab."""
    def apply(self, img: np.ndarray, **params):
        img = color.rgb2lab(img)
        return img

class RGB2YCbCr(ImageOnlyTransform):
    """Convert ndarray image to ndarray YCbCr."""
    def apply(self, img: np.ndarray, **params):
        img = color.rgb2ycbcr(img)
        return img