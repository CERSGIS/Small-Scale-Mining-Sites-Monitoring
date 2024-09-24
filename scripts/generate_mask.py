import numpy as np
# import pandas as pd
import image_utils as iu
import tensorflow as tf
import os
from PIL import Image, ImageEnhance
# from PIL import ImageFilter
from keras.models import load_model
# from shutil import copy2
# import matplotlib.pyplot as plt
import numpy_utils as nu
import geotiff_utils as gu
from osgeo import gdal


def real_image_preprocessing_func(np_gt):
    np_gt = np.clip(np_gt, None, 6000)
    return np_gt / 6000

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# Predicts the mask of an image.
def predict_mask(unet, np_img, input_shape):
    # split image
    # image_width, image_height = input_shape
    height, width = np_img.shape[0], np_img.shape[1]
    target_height, target_width = input_shape
    if np_img.shape[0] != target_height or np_img.shape[1] != target_width:
        np_img = nu.zero_pad_array(np_img, input_shape)
    # np_img = nu.resize_np_image(np_img, (image_height, image_width))
    np_img = real_image_preprocessing_func(np_img)
    pred = unet.predict(np.array([np_img]), verbose=False)
    np_mask = np.array(create_mask(pred), dtype=np.uint8)
    np_mask = nu.crop_np_array(np_mask, 0, 0, width, height)
    # print(np_mask.max())
    return np_mask


def pre_process_func(dest_gt, model, max_grid_shape):
    def predict_grid(np_gt, extent, image_coord, progress):
        # if int(np_gt.max()) > 0:
        mask_np_gt = predict_mask(model, np_gt, max_grid_shape)
        position = image_coord[0], image_coord[1]
        # print('position' ,position)
        gu.write_geoTiff_bands(dest_gt, mask_np_gt, position)

    return predict_grid


gt_dir_path = input('Source path:')
dest_path = input('Destination path:')
mc_unet = load_model('multi_class_unet_model_2023-10-13 10-08-17.294990.h5', compile=False)
# gu.crop_and_process_geoTiff(gt, pre_process_func(dest_gt, mc_unet, max_grid_shape), max_grid_shape, 'uint16')
gt_paths = iu.get_all_files(
    gt_dir_path, '*.tif', True)
# dest_path = os.path.join(data_path, prediction_path, '2016')
iu.create_dir_if_not_exists(dest_path)
for gt_path in gt_paths:
    # try:
    dest_file = os.path.join(dest_path, iu.get_file_or_foldername(gt_path))
    if os.path.exists(dest_file):
        print('skipping', gt_path)
        continue
    print('Processing', gt_path)
    gt = gdal.Open(gt_path)
    proj = gt.GetProjection()
    size = gt.RasterXSize, gt.RasterYSize
    geo_transform = gt.GetGeoTransform()
    dest_gt = gu.create_new_geoTiff(dest_file, size, 1, proj, geo_transform, compression='None')
    g_gt = gt
    dest_gt = dest_gt
    # print(dest_gt)
    # print(gt)
    max_grid_shape = (1024, 1024)
    gu.crop_and_process_geoTiff(gt, pre_process_func(dest_gt, mc_unet, max_grid_shape), max_grid_shape, 'uint16')
    gt = None
    dest_gt = None
    print(dest_file, 'saved')
print('Complete')