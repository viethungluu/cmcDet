import numpy as np
from skimage import color
from PIL import Image

from albumentations.core.transforms_interface import ImageOnlyTransform

class RGB2Lab(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2lab(image)
        return image

class RGB2HSV(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray HSV."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2hsv(image)
        return image


class RGB2HED(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray HED."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2hed(image)
        return image


class RGB2LUV(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray LUV."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2luv(image)
        return image


class RGB2YUV(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray YUV."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2yuv(image)
        return image


class RGB2XYZ(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray XYZ."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2xyz(image)
        return image


class RGB2YCbCr(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray YCbCr."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2ycbcr(image)
        return image


class RGB2YDbDr(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2ydbdr(image)
        return image


class RGB2YPbPr(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray YPbPr."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2ypbpr(image)
        return image


class RGB2YIQ(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray YIQ."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2yiq(image)
        return image


class RGB2CIERGB(ImageOnlyTransform):
    """Convert RGB PIL image to ndarray RGBCIE."""
    def __call__(self, image: Image, **kwargs):
        image = np.asarray(image, np.uint8)
        image = color.rgb2rgbcie(image)
        return image
