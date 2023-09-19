import numpy as np
from PIL import Image, ImageEnhance

def flip_np_image_left_right(np_image):
    # Flip the array from left to right
    return np.fliplr(np_image)


def rotate_np_image(np_image, angle):
    return np.rot90(np_image, k=int((360 - angle) / 90))


def resize_np_image(np_image, dimension):
    width, height = dimension
    no_channels = np_image.shape[2]
    output = np.zeros(
        (int(height), int(width), no_channels), str(np_image.dtype))
    for x in range(no_channels):
        channel = np.array(np_image[:, :, x], str(np_image.dtype))
        channel_image = Image.fromarray(channel)
        channel_image = channel_image.resize((width, height))
        channel = np.asarray(channel_image)
        output[..., x] = channel
    return output


def load_numpy_file(path):
    loaded = np.load(path, allow_pickle=True)
    return loaded['array1']


def save_numpy_file(np_arr, path):
    np.savez(path, array1=np_arr)

def count_value(array, value):
    """
    Count the number of times a value appears in a NumPy array.

    Parameters:
    - array (numpy.ndarray): Input NumPy array.
    - value: Value to count occurrences of.

    Returns:
    - count (int): Number of occurrences of the value in the array.
    """
    count = np.count_nonzero(array == value)
    return count