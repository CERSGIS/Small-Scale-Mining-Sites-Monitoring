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
 
def crop_np_array(arr, left, top, right, bottom):
    """
    Crop the input NumPy array to the specified bounding box.

    Args:
        arr (numpy.ndarray): Input array to be cropped.
        left (int): Left coordinate of the bounding box.
        top (int): Top coordinate of the bounding box.
        right (int): Right coordinate of the bounding box.
        bottom (int): Bottom coordinate of the bounding box.

    Returns:
        numpy.ndarray: Cropped array within the specified bounding box.

    Raises:
        ValueError: If the bounding box coordinates are invalid or exceed the array size.
    """
    # Validate the bounding box coordinates
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError("Invalid bounding box coordinates.")

    # Get the dimensions of the input array
    arr_height, arr_width = arr.shape[:2]

    # Check if the bounding box exceeds the array size
    if right > arr_width or bottom > arr_height:
        raise ValueError("Bounding box exceeds the array size.")

    # Crop the array to the specified bounding box
    cropped_arr = arr[top:bottom, left:right]

    return cropped_arr

def zero_pad_array(input_array, new_shape):
    """
    Zero-pad the input_array to the specified new_shape.

    Args:
        input_array (numpy.ndarray): Input array of shape (height, width, x).
        new_shape (tuple): Desired new shape (new_height, new_width, x).

    Returns:
        numpy.ndarray: Zero-padded array of shape (new_height, new_width, x).
    """
    if input_array.shape[0] >= new_shape[0] and input_array.shape[1] >= new_shape[1]:
        return input_array
    # Calculate the amount of padding needed for each dimension
    pad_height = new_shape[0] - input_array.shape[0]
    pad_width = new_shape[1] - input_array.shape[1]

    # Create a tuple of padding values for each dimension
    pad_values = [(0, pad_height), (0, pad_width)] + [(0, 0)] * (input_array.ndim - 2)

    # Use numpy.pad to pad the array with zeros
    zero_padded_array = np.pad(input_array, pad_values, mode='constant', constant_values=0)

    return zero_padded_array

def normalize_np_array(array, min, max):
    """
    Normalize an array to a range between 0.0 and 1.0.

    Args:
        array: The array to be normalized.
        min: The minimum value in the array.
        max: The maximum value in the array.

    Returns:
        A normalized array with values between 0.0 and 1.0.
    """
    return (array - min) / (max - min)

