import numpy as np
from skimage import color
from PIL import Image

def rgb_to_lab(rgb_image: Image):
    # Convert PIL image to NumPy array
    rgb_array = np.array(rgb_image)

    # Convert RGB to LAB
    lab_array = color.rgb2lab(rgb_array)

    # Create a new PIL image from the LAB array
    lab_image = Image.fromarray(lab_array.astype('uint8'))

    return lab_image


if __name__ == '__main__':
    import sys    

    # Example usage
    input_image_path = sys.argv[1]

    # Open the image using PIL
    rgb_image = Image.open(input_image_path).convert('RGB')

    lab_image = rgb_to_lab(rgb_image)

    # Separate LAB channels
    lab_array = np.array(lab_image)
    l_channel = lab_array[:, :, 0]
    ab_channels = lab_array[:, :, 1:]

    import matplotlib.pyplot as plt
    # Visualize L channel
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(l_channel, cmap='gray')
    plt.title('L Channel')
    plt.axis('off')

    # Visualize ab channels
    plt.subplot(1, 2, 2)
    ab_channels_rescaled = (ab_channels + [0, 128]) * [255/100, 255/128]
    ab_image_rgb = np.concatenate((np.zeros_like(l_channel)[:, :, np.newaxis], ab_channels_rescaled), axis=2)
    ab_image_rgb = color.lab2rgb(ab_image_rgb)
    plt.imshow(ab_image_rgb)
    plt.imshow(ab_image_rgb, cmap='cividis')  # You can choose any valid Matplotlib colormap
    plt.title('ab Channels')
    plt.axis('off')

    plt.show()

    