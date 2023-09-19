from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pathlib
from sklearn.metrics import precision_score, recall_score

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(path, color_space=None):
    if color_space is None:
        img = Image.open(path)
    else:
        img = Image.open(path).convert(color_space)
    return img


def display_image(image, cmap='brg'):
    fig = plt.figure
    plt.imshow(image, cmap=cmap)
    
def display_images(image_dict, grid):
    if not isinstance(image_dict, dict):
        raise ValueError("The 'image_dict' parameter must be a dictionary.")
    rows, cols = grid
    num_images = len(image_dict)
    if rows is None and cols is None:
        rows = int(num_images ** 0.5)
        cols = int(num_images ** 0.5)
    elif rows is None:
        rows = (num_images + cols - 1) // cols
    elif cols is None:
        cols = (num_images + rows - 1) // rows
    fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
    labels = list(image_dict.keys())
    images = list(image_dict.values())

    for i, ax in enumerate(axs.flat):
        if i < num_images:
            if isinstance(images[i], tuple):
               image, cmap = images[i]
               ax.imshow(image, cmap)
            else:
               ax.imshow(images[i])
            ax.axis('off')
            ax.set_title(labels[i])
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    

# def display_images(images, labels=None, row_count=10, col_count=10, cmap='brg'):
#     num = row_count * col_count
#     fig, axes = plt.subplots(row_count, col_count,
#                              figsize=(1.5 * row_count, 2 * col_count))
#     for i in range(len(images)):
#         ax = axes[i // row_count, i % col_count]
#         ax.imshow(images[i], cmap=cmap)
#         if labels != None:
#             ax.set_title('Label: {}'.format(labels[i]))
#     plt.tight_layout()
#     plt.show()


def load_images_in_dir(path, color_space):
    images_path = get_all_files(path)
    images = [None] * len(images_path)
    for x in range(len(images_path)):
        images[x] = read_image(os.path.join(
            path, images_path[x]), color_space)
    return images


def create_dir_if_not_exists(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


# def get_all_files(path, search ,get_full_path=False):
#     if get_full_path:
#         onlyfiles = [path + '/' + f for f in os.listdir(
#             path) if os.path.isfile(os.path.join(path, f))]
#     else:
#         onlyfiles = [f for f in os.listdir(
#             path) if os.path.isfile(os.path.join(path, f))]
#     return onlyfiles

def get_all_files(path, pattern='*', get_full_path=False):
    files = list(pathlib.Path(path).glob(pattern))
    if get_full_path:
        onlyfiles = [os.path.join(path, f.name) for f in files if f.is_file()]
    else:
        onlyfiles = [f.name for f in files if f.is_file()]
    return onlyfiles


def get_new_image_dimen(image, new_dimen):
    width = image.width
    height = image.height
    new_width = 0
    new_height = 0
    aspect_ratio = width / height
    if width > height:
        new_width = new_dimen
        new_height = new_dimen / aspect_ratio
    else:
        new_height = new_dimen
        new_width = new_dimen * aspect_ratio
    return (new_width, new_height)


def pad_image(image, max_size):
    old_image_height, old_image_width, channels = image.shape
    # create new image of desired size and color (blue) for padding
    new_image_width = max_size
    new_image_height = max_size
    color = (0, 0, 0)
    result = np.full((new_image_height, new_image_width,
                      channels), color, dtype=np.uint8)
    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    # print(image)
    # copy img image into center of result image
    # result[y_center:y_center+old_image_height,x_center:x_center+old_image_width] = image
    result[0:old_image_height, 0:old_image_width] = image
    return result


def rgb2ycbcr(im):
    xform = np.array(
        [[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def set_image_contrast(image, factor):
    # image brightness enhancer
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def rotate_image(image, angle):
    angle = 360 - angle
    image = image.rotate(angle)

# def calc_new_bbox(bbox, angle):

# def read_bbox_from_xml(filename):


def plot_rect(image, rect, box_color):
    x, y, w, h = rect
    r, g, b = box_color
    np_image = np.asarray(image)
    # get the row to mark.
    np_image[y, x: x + w] = [r, g, b]
    # np_image[y : y + y, x] = [r, g, b]
    return Image.fromarray(np_image)
    # np_image[x : x+ h, ]


def print_progress(current, total):
    # Calculate the percentage of progress
    progress = (current / total) * 100
    # Print the progress bar
    print("\rProgress: [{0:50s}] {1:.1f}%".format(
        '#' * int(progress/2), progress), end="")


def get_subdirectories(path, recursive=False):
    subdirectories = []
    for item in os.scandir(path):
        if item.is_dir():
            subdirectories.append(item.path)
            if recursive:
                subdirectories += get_subdirectories(item.path, recursive)
    return subdirectories


def get_file_or_foldername(path):
    return os.path.basename(path)

# Splits image into specified rows and columns


def split_image(image, row_count, col_count):
    parts = []
    width, height = image.size
    left = 0
    top = 0
    right = width / col_count
    bottom = height / row_count
    for r in range(row_count):
        top = int(r * (height / row_count))
        bottom = int(top + (height / row_count))
        for c in range(col_count):
            left = int(c * (width / col_count))
            right = int(left + (width / col_count))
            part = image.crop((left, top, right, bottom))
            parts.append(part)
    return parts


def calculate_precision_recall_all_classes(data_generator, model, num_classes):
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    total_samples = 0
    for images, ground_truth_masks in data_generator:
        # Perform predictions using the trained model
        predicted_masks = np.argmax(model.predict(images), axis=-1)
        for class_id in range(num_classes):
            true_positives = np.sum((ground_truth_masks == class_id) & (predicted_masks == class_id))
            false_positives = np.sum((ground_truth_masks != class_id) & (predicted_masks == class_id))
            false_negatives = np.sum((ground_truth_masks == class_id) & (predicted_masks != class_id))

            precision[class_id] += true_positives / (true_positives + false_positives)
            recall[class_id] += true_positives / (true_positives + false_negatives)
        total_samples += images.shape[0]
    precision /= total_samples
    recall /= total_samples
    class_values = np.arange(num_classes)
    df = pd.DataFrame({'Class_value': class_values, 'Precision': precision, 'Recall': recall})
    return df

'''Gets the intersection between two bounding boxes'''
def calculate_intersection_bbox(bbox_A, bbox_B):
    xmin_A, xmax_A, ymin_A, ymax_A = bbox_A
    xmin_B, xmax_B, ymin_B, ymax_B = bbox_B
    x_decrease = False
    y_decrease = False
    # Check if the x-coordinate is in decreasing order
    if xmin_A > xmax_A:
        x_decrease = True
        xmin_A, xmax_A = xmax_A, xmin_A
    if xmin_B > xmax_B:
        xmin_B, xmax_B = xmax_B, xmin_B
    # Check if the y-coordinate is in decreasing order
    if ymin_A > ymax_A:
        y_decrease = True
        ymin_A, ymax_A = ymax_A, ymin_A
    if ymin_B > ymax_B:
        ymin_B, ymax_B = ymax_B, ymin_B

    # Calculate the intersection coordinates
    xmin_intersect = max(xmin_A, xmin_B)
    xmax_intersect = min(xmax_A, xmax_B)
    ymin_intersect = max(ymin_A, ymin_B)
    ymax_intersect = min(ymax_A, ymax_B)

    # Check if there is a valid intersection
    if xmin_intersect < xmax_intersect and ymin_intersect < ymax_intersect:
        # Swap the x-coordinate and xmax/xmin if the resolution is negative
        if x_decrease:
            xmin_intersect, xmax_intersect = xmax_intersect, xmin_intersect
        # Swap the y-coordinate and ymax/ymin if the resolution is negative
        if y_decrease:
            ymin_intersect, ymax_intersect = ymax_intersect, ymin_intersect
            # ymin_intersect, ymax_intersect = ymax_intersect, ymin_intersect

        return (xmin_intersect, xmax_intersect, ymin_intersect, ymax_intersect)
    else:
        # No intersection exists
        return None
