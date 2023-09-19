from osgeo import gdal, ogr, osr
from osgeo import gdal, gdalconst, osr
# from rasterio.mask import mask
# from rasterio.features import geometry_mask
# from shapely.geometry import shape, box
#import rasterio
from scipy.ndimage import zoom
from osgeo import gdal
# from osgeo import gdal_array
from osgeo import osr

import numpy as np
# import pandas as pd
# import image_utils as iu
# import tensorflow as tf
import os
from PIL import Image, ImageEnhance
# from PIL import ImageFilter
# from keras.models import load_model
# from shutil import copy2
# import matplotlib.pyplot as plt


def get_rgb_bands(tiff, color_depth='uint8'):
    red = tiff.GetRasterBand(1).ReadAsArray()
    green = tiff.GetRasterBand(2).ReadAsArray()
    blue = tiff.GetRasterBand(3).ReadAsArray()
    # rgbOutput = source.ReadAsArray() #Easier method
    rgbOutput = np.zeros((tiff.RasterYSize, tiff.RasterXSize, 3), color_depth)
    rgbOutput[..., 0] = red
    rgbOutput[..., 1] = green
    rgbOutput[..., 2] = blue
    # Clear so file isn't locked
    source = None
    return rgbOutput


def get_l_band(tiff, band, color_depth='uint8'):
    L = tiff.GetRasterBand(band).ReadAsArray()
    # rgbOutput = source.ReadAsArray() #Easier method
    rgbOutput = np.zeros((tiff.RasterYSize, tiff.RasterXSize, 1), color_depth)
    rgbOutput[..., 0] = L
    # Clear so file isn't locked
    source = None
    return rgbOutput


def save_image_as_geotiff(mask, destination_path, dataset):
    mask = mask.resize((dataset.RasterXSize, dataset.RasterYSize))
    array = np.asarray(mask)
    nrows, ncols = array.shape[0], array.shape[1]
    depth = array.shape[2]
    # geotransform = (0, 1024, 0, 0, 0, 512)
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    # I don't know why rotation is in twice???
    output_raster = gdal.GetDriverByName('GTiff').Create(
        destination_path, ncols, nrows, depth, gdal.GDT_Byte)  # Open the file
    # print(output_raster)
    output_raster.SetGeoTransform(
        dataset.GetGeoTransform())  # Specify its coordinates
    output_raster.SetProjection(dataset.GetProjection())
    output_raster.SetMetadata(output_raster.GetMetadata())
   # to the file
    for x in range(depth):
        output_raster.GetRasterBand(x + 1).WriteArray(
            array[:, :, x])   # Writes my array to the raster
    output_raster.FlushCache()


def save_np_array_as_geoTiff(np_array, destination_path, output_dtype=gdal.GDT_UInt16, geoTransform=None, projection=None, metadata=None):
    # np_array = np_array.resize((dataset.RasterXSize, dataset.RasterYSize))
    array = np_array
    height, width = array.shape[0], array.shape[1]
    depth = array.shape[2]
    # geotransform = (0, 1024, 0, 0, 0, 512)
    # That's (top left x, w-e pixel resolution, rotation (0 if North is up),
    #         top left y, rotation (0 if North is up), n-s pixel resolution)
    # I don't know why rotation is in twice???
    output_raster = gdal.GetDriverByName('GTiff').Create(
        destination_path, width, height, depth, output_dtype)  # Open the file
    # print(output_raster)
    if geoTransform:
        output_raster.SetGeoTransform(geoTransform)  # Specify its coordinates
    if projection:
        output_raster.SetProjection(projection)
    if metadata:
        output_raster.SetMetadata(metadata)
    for x in range(depth):
        output_raster.GetRasterBand(x + 1).WriteArray(
            array[:, :, x])   # Writes my array to the raster
    output_raster.FlushCache()
    return output_raster


def read_geoTiff_bands(geoTiff, bands=None, bbox=None, dtype='uint16'):
    width = geoTiff.RasterXSize
    height = geoTiff.RasterYSize
    if bbox == None:
       bbox = (0, width, 0, height)
    left, right, top, bottom = bbox
    bb_width = right - left
    bb_height = bottom - top
    band_list = []
    if bands != None:
        for val in bands:
            band = geoTiff.GetRasterBand(val).ReadAsArray(left, top,
                                                          int(bb_width), int(bb_height))
            band_list.append(band)
    else:
        for x in range(geoTiff.RasterCount):
            band = geoTiff.GetRasterBand(x + 1).ReadAsArray(left, top,
                                                            int(bb_width), int(bb_height))
            band_list.append(band)
    output = np.zeros(
        (int(bb_height), int(bb_width), len(band_list)), dtype)
    for x in range(len(band_list)):
        output[..., x] = band_list[x]
    return output


def write_geoTiff_bands(output_raster, np_array, position):
    """
    Write a numpy array to a GeoTIFF file.

    Parameters:
    - output_raster (gdal.Dataset): Output GeoTIFF dataset.
    - np_array (numpy.ndarray): Numpy array to be written.
    - position (tuple): Top-left position (left, top) to write the array in the GeoTIFF.
    """
    array = np_array
    left, top = position
    height, width = array.shape[0], array.shape[1]
    bands = array.shape[2]
    for x in range(bands):
        output_raster.GetRasterBand(x + 1).WriteArray(
            array[:, :, x], xoff=left, yoff=top)   # Writes my array to the raster
    output_raster.FlushCache()


def display_np_geoTiff(np_geoTiff, rgb_bands):
    r, g, b = rgb_bands
    red_band = np_geoTiff[:, :, r]
    green_band = np_geoTiff[:, :, g]
    blue_band = np_geoTiff[:, :, b]
    shape = np_geoTiff.shape
    # max = np_geoTiff.max()
    # np_rgb_image = np_geoTiff[:, :, 0:3][:, :, ::-1]
    rgbOutput = np.zeros((shape[0], shape[1], 3), dtype=np_geoTiff.dtype)
    rgbOutput[..., 0] = red_band
    rgbOutput[..., 1] = green_band
    rgbOutput[..., 2] = blue_band
    rgbOutput = (rgbOutput / rgbOutput.max()) * 255
    im = Image.fromarray(np.array(rgbOutput, dtype=np.uint8))
    return im


def display_np_geoTiff_band(band_geoTiff, band_number):
    np_gray_image = band_geoTiff[:, :, band_number]
    max = np_gray_image.max()
    if max != 0:
        np_n_im = (np_gray_image / max) * 255
    else:
        np_n_im = np_gray_image
    im = Image.fromarray(np.array(np_n_im, dtype=np.uint8))
    return im


# def crop_geoTiff(geoTiff, left, top, right, bottom, dtype='uint16'):
#     width = abs(right - left)
#     height = abs(top - bottom)
#     bands = [None] * geoTiff.RasterCount
#     for x in range(geoTiff.RasterCount):
#         band = geoTiff.GetRasterBand(x + 1).ReadAsArray(left, top,
#                                                         int(width), int(height))
#         bands[x] = band
#     output = np.zeros(
#         (int(height), int(width), geoTiff.RasterCount), dtype)
#     for x in range(len(bands)):
#         output[..., x] = bands[x]
#     return output
def crop_geoTiff(geoTiff, left, top, right, bottom, dtype=None):
    """
    Crop a GeoTIFF array to the specified region.

    Parameters:
    - geoTiff (gdal.Dataset): Input GeoTIFF dataset.
    - left (int): Left coordinate of the crop region.
    - top (int): Top coordinate of the crop region.
    - right (int): Right coordinate of the crop region.
    - bottom (int): Bottom coordinate of the crop region.
    - dtype (str or None, optional): Desired data type of the output array. 
    If set to None, the data type is inferred from the band. Defaults to None.

    Returns:
    - output (numpy.ndarray): Cropped array with dimensions (height, width, bands).

    Raises:
    - ValueError: If the crop dimensions exceed the size of the GeoTIFF.
    """
    if (int(right) > geoTiff.RasterXSize) or (int(bottom) > geoTiff.RasterYSize):
        # print(right, bottom)
        # print(geoTiff.RasterXSize, geoTiff.RasterYSize)
        raise ValueError('Crop dimensions exceed the size of the GeoTIFF.')
    if dtype is None:
        dtype = get_geoTiff_numpy_datatype(geoTiff)
    width = abs(right - left)
    height = abs(top - bottom)
    output = np.zeros(
        (int(height), int(width), geoTiff.RasterCount), dtype)
    # bands = [None] * geoTiff.RasterCount
    for x in range(geoTiff.RasterCount):
        band = geoTiff.GetRasterBand(x + 1).ReadAsArray(left, top,
                                                        int(width), int(height))
        output[..., x] = band
    return output


def get_geoTiff_part(geoTiff, left, top, right, bottom, dtype='uint16'):
    c_gt = crop_geoTiff(geoTiff, left, top, right, bottom, dtype)
    extent = get_geoTiff_extent(geoTiff, (left, right, top, bottom))
    return c_gt, extent

# Splits image into specified rows and columns


def split_geoTiff_image(geoTiff, row_count, col_count):
    parts = []
    width, height = geoTiff.RasterXSize, geoTiff.RasterYSize
    left = 0
    top = 0
    # right = width / col_count
    # bottom = height / row_count
    # bands = read_geoTiff_bands(geoTiff)
    for r in range(row_count):
        top = int(r * (height / row_count))
        bottom = int(top + (height / row_count))
        for c in range(col_count):
            left = int(c * (width / col_count))
            right = int(left + (width / col_count))
            part = crop_geoTiff(geoTiff, left, top, right, bottom)
            parts.append(part)
    return parts


def crop_geoTiff_into_grids(geoTiff, max_grid_shape, yield_results=False, dtype = 'uint16'):
    """
    Split a GeoTIFF into grids of numpy arrays with a specified maximum shape.

    Parameters:
    - geoTiff (gdal.Dataset): Input GeoTIFF dataset.
    - max_grid_shape (tuple): Maximum shape of each grid (rows, columns).
    - yield_results (bool, optional): Whether to yield each grid individually or return all grids in a list. 
                                      Defaults to True (yield results).

    Yields or Returns:
    - grid (numpy.ndarray): Numpy array representing a divided grid.
      (Yielded if yield_results=True, Returned as a list if yield_results=False)

    Raises:
    - ValueError: If the max_grid_shape is invalid or exceeds the size of the GeoTIFF.
    """
    if not isinstance(max_grid_shape, tuple) or len(max_grid_shape) != 2 or \
       max_grid_shape[0] <= 0 or max_grid_shape[1] <= 0:
        raise ValueError(
            "Invalid max_grid_shape. It should be a tuple of two positive integers.")

    if max_grid_shape[0] > geoTiff.RasterYSize or max_grid_shape[1] > geoTiff.RasterXSize:
        raise ValueError("max_grid_shape exceeds the size of the GeoTIFF.")

    rows, cols = geoTiff.RasterYSize, geoTiff.RasterXSize
    max_grid_height, max_grid_width = max_grid_shape
    total_grids = np.ceil(rows / max_grid_height) * np.ceil(cols / max_grid_width)
    if yield_results:
        progress = 0
        for r in range(0, rows, max_grid_height):
            for c in range(0, cols, max_grid_width):
                left, top = c, r
                right, bottom = min(
                    c + max_grid_width, cols), min(r + max_grid_height, rows)

                grid, extent = get_geoTiff_part(
                    geoTiff, left, top, right, bottom, dtype)
                progress += 1
                yield (grid, extent, (left, top, right, bottom), (progress, total_grids))
    else:
        grids = []
        for r in range(0, rows, max_grid_height):
            for c in range(0, cols, max_grid_width):
                left, top = c, r
                right, bottom = min(
                    c + max_grid_width, cols), min(r + max_grid_height, rows)

                grid, extent = get_geoTiff_part(
                    geoTiff, left, top, right, bottom)
                grids.append((grid, extent, (left, top, right, bottom)))
        return grids


def resize_np_image(np_image, size):
    width, height = size
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


def resize_array(array, new_size):
    width, height = new_size
    new_shape = (height, width)
    resized_array = zoom(
        array, (new_shape[0] / array.shape[0], new_shape[1] / array.shape[1], 1))
    return resized_array


def create_new_geoTiff(dest_path, size, band_count, projection, geo_transform, dtype=gdal.GDT_Byte, compression='DEFLATE', zlevel=9):
    """
    Create a new GeoTIFF file with the specified dimensions, bands, projection, and geotransformation.

    Parameters:
        - dest_path (str): The path where the new GeoTIFF file will be created.
        - size (tuple): A tuple specifying the width and height of the new GeoTIFF in pixels. (width, height)
        - band_count (int): The number of bands in the new GeoTIFF.
        - projection (str): The projection of the GeoTIFF file in Well-Known Text (WKT) format.
        - geo_transform (tuple): A tuple representing the geotransformation parameters for the GeoTIFF.
                               (originX, pixelWidth, 0, originY, 0, pixelHeight)
        - dtype (int, optional): The data type of the pixel values in the new GeoTIFF. Default is gdal.GDT_Byte (8-bit).
        - compression (str, optional): Compression method for the GeoTIFF. Default is 'DEFLATE'.
        - zlevel (int, optional): Compression level for the GeoTIFF. Default is 5.
    Returns:
        gdal.Dataset: The GDAL dataset representing the newly created GeoTIFF file.
    """

    # Create the GeoTIFF file with the specified dimensions
    width, height = size
    driver = gdal.GetDriverByName("GTiff")
    # Set compression options
    options = ['COMPRESS=' + compression, 'ZLEVEL=' + str(zlevel)]
    ds = driver.Create(dest_path, width, height,
                       band_count, dtype, options=options)
    # Define the projection of the file
    ds.SetProjection(projection)
    # Specify its coordinates
    ds.SetGeoTransform(geo_transform)
    return ds


def merge_geoTiffs(geoTiff_paths, output_path, compression='NONE'):
    """
    Merge multiple GeoTIFF files into a single GeoTIFF file.

    Parameters:
        geoTiff_paths (list): A list of file paths to the input GeoTIFF files that need to be merged.
        output_path (str): The path where the merged GeoTIFF file will be created.
        compression (str, optional): The compression method to be used in the output GeoTIFF file.
                                     Default is 'NONE'. Other options include 'LZW', 'DEFLATE', etc.

    Returns:
        None
    """
    # Initialize a list to store the GDAL datasets of input GeoTIFF files
    geoTiffs = []
    # Open each input GeoTIFF file and add the GDAL dataset to the list
    for geoTiff_path in geoTiff_paths:
        geoTiff = gdal.Open(geoTiff_path, gdal.GA_ReadOnly)
        geoTiffs.append(geoTiff)
    # Set up the warp options for merging
    warp_options = gdal.WarpOptions(format='GTiff', options=[
                                    'COMPRESS=' + compression])
    # Merge the GeoTIFF files using gdal.Warp and save the merged file to the output path
    g = gdal.Warp(output_path, geoTiffs, format="GTiff", options=warp_options)
    # Close the GDAL dataset to release resources
    g = None


def get_geoTiff_datatype(geoTiff):
    """
    Get the GDAL data type of a GeoTIFF dataset.

    Parameters:
    - geoTiff (gdal.Dataset): Input GeoTIFF dataset.

    Returns:
    - int: GDAL data type of the GeoTIFF dataset.

    """
    band = geoTiff.GetRasterBand(1)
    return band.DataType


def get_geoTiff_numpy_datatype(geoTiff):
    """
    Get the NumPy data type string of a GeoTIFF dataset.

    Parameters:
    - geoTiff (gdal.Dataset): Input GeoTIFF dataset.

    Returns:
    - str: NumPy data type string of the GeoTIFF dataset.

    """
    gt_dtype = get_geoTiff_datatype(geoTiff)
    gdal_to_numpy_datatype = {
        gdal.GDT_Byte: 'uint8',
        gdal.GDT_UInt16: 'uint16',
        gdal.GDT_Int16: 'int16',
        gdal.GDT_UInt32: 'uint32',
        gdal.GDT_Int32: 'int32',
        gdal.GDT_Float32: 'float32',
        gdal.GDT_Float64: 'float64'
    }
    numpy_datatype = gdal_to_numpy_datatype.get(gt_dtype, None)
    return numpy_datatype


def get_gdal_datatype_from_numpy_datatype(dtype: str):
    numpy_dtype_to_gdal = {
        'uint8': gdal.GDT_Byte,
        'uint16': gdal.GDT_UInt16,
        'int16': gdal.GDT_Int16,
        'uint32': gdal.GDT_UInt32,
        'int32': gdal.GDT_Int32,
        'float32': gdal.GDT_Float32,
        'float64': gdal.GDT_Float64
    }
    gdal_dtype = numpy_dtype_to_gdal[dtype]
    return gdal_dtype


def get_geoTiff_extent(geoTiff, image_bounds=None):
    """
    Get the geographic extent (bounding box) of a GeoTIFF.

    Parameters:
    - geoTiff (GDAL Dataset): Input GeoTIFF dataset object.
    - image_bounds (tuple, optional): Bounds of the specific area of interest within the GeoTIFF.
                                      Format: (xmin, xmax, ymin, ymax). Default is None,
                                      which represents the entire extent of the GeoTIFF.

    Returns:
    - tuple: Geographic extent (left, right, top, bottom) in the coordinate reference system of the GeoTIFF.

    Note:
    - The function assumes the input GeoTIFF is in a projected coordinate reference system (CRS).
    - If image_bounds is provided, the extent will be calculated based on the specified bounds.
      Otherwise, the entire extent of the GeoTIFF will be used.
    - The returned extent represents the geographic coordinates (left, right, top, bottom)
      within the CRS of the GeoTIFF.
    
    Exceptions:
    - ValueError: Raised if the image bounds are outside the valid range of the GeoTIFF.

    """
    xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i = geoTiff.GetGeoTransform()
    width, height = geoTiff.RasterXSize, geoTiff.RasterYSize
    if image_bounds is None:
        xmin, xmax, ymin, ymax = 0, geoTiff.RasterXSize, 0, geoTiff.RasterYSize
    else:
        xmin, xmax, ymin, ymax = image_bounds
        if xmin < 0 or xmax > width or ymin < 0 or ymax > height:
            raise ValueError(
                "Image bounds are outside the valid range of the GeoTIFF.")
    left = xmin_i + (xmin * xres_i)
    right = xmin_i + (xmax * xres_i)
    top = ymin_i + (ymin * yres_i)
    bottom = ymin_i + (ymax * yres_i)
    return (left, right, top, bottom)

# def get_geoTiff_extent(geoTiff, image_bounds):
#     if image_bounds is not None:
#            xmin, xmax, ymin, ymax = image_bounds
#     else:
#        xmin, xmax ,ymin, ymax =0, geoTiff.RasterXSize, 0, geoTiff.RasterYSize
#     xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i = geoTiff.GetGeoTransform()
#     left = xmin_i + (xmin * xres_i)
#     right = xmin_i + (xmax * xres_i)
#     top = ymin_i + (ymin * yres_i)
#     bottom = ymin_i + (ymax * yres_i)
#     return (left, xres_i, xskew_i, right, top, bottom)


def get_geoTiff_image_coordinates(geoTiff, extent):
    """
    Get the image coordinates corresponding to a given geographic extent in a GeoTIFF.

    Parameters:
    - geoTiff (GDAL Dataset): Input GeoTIFF dataset object.
    - extent (tuple): Geographic extent (left, right, top, bottom) in the coordinate reference system of the GeoTIFF.

    Returns:
    - tuple: Image coordinates (left, right, top, bottom) corresponding to the provided geographic extent.

    Note:
    - The function assumes the input GeoTIFF is in a projected coordinate reference system (CRS).
    - The extent represents the geographic coordinates (left, right, top, bottom) within the CRS of the GeoTIFF.
    - The returned image coordinates represent the corresponding pixel coordinates (xmin, xmax, ymin, ymax) within the GeoTIFF.
    """
    xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i = geoTiff.GetGeoTransform()
    width, height = geoTiff.RasterXSize, geoTiff.RasterYSize
    e_xmin, e_xmax, e_ymin, e_ymax = extent

    # if e_xmin < xmin_i or e_xmax > (xmin_i + width * xres_i) or e_ymin < ymin_i or e_ymax > (ymin_i + height * yres_i):
    #     raise ValueError("Geographic extent is outside the valid range of the GeoTIFF.")

    left = (e_xmin - xmin_i) / xres_i
    right = (e_xmax - xmin_i) / xres_i
    top = (e_ymin - ymin_i) / yres_i
    bottom = (e_ymax - ymin_i) / yres_i
    return left, right, top, bottom


def calculate_geoTiff_intersection_extent(geoTiff_1, geoTiff_2):
    """
    Calculates the intersection extent between two GeoTIFFs.

    Parameters:
    - geoTiff_1 (GDAL Dataset): First input GeoTIFF dataset object.
    - geoTiff_2 (GDAL Dataset): Second input GeoTIFF dataset object.

    Returns:
    - tuple: Intersection extent (left, right, top, bottom) in the coordinate reference system of the GeoTIFFs or None if no intersection.

    Exceptions:
    - ValueError: Raised if the GeoTIFFs have different projections.

    """
    # Check if both GeoTIFFs have the same projection
    projection_1 = geoTiff_1.GetProjection()
    projection_2 = geoTiff_2.GetProjection()
    if projection_1 != projection_2:
        raise ValueError("The GeoTIFFs have different projections.")
    extent1 = get_geoTiff_extent(geoTiff_1)
    extent2 = get_geoTiff_extent(geoTiff_2)
    intersection_extent = calculate_intersection_bbox(extent1, extent2)
    return intersection_extent


def calculate_intersection_bbox(bbox_A, bbox_B):
    """
    Calculate the intersection bounding box between two bounding boxes.

    Parameters:
    - bbox_A (tuple): Bounding box A in the format (xmin, xmax, ymin, ymax).
    - bbox_B (tuple): Bounding box B in the format (xmin, xmax, ymin, ymax).

    Returns:
    - tuple or None: Intersection bounding box in the format (xmin, xmax, ymin, ymax).
                     Returns None if no intersection exists.

    """
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


def reproject_geoTiff(input_file_path, target_projection, output_file_path, compression='DEFLATE', zlevel=5, output_dtype=gdal.GDT_UInt16):
    # Open the input GeoTIFF
    input_ds = gdal.Open(input_file_path, gdalconst.GA_ReadOnly)
    if input_ds is None:
        raise IOError(f"Failed to open input file: {input_file_path}")

    # Get the input projection
    input_projection = input_ds.GetProjection()

    # Create a transformation object for the reprojection
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(target_projection)
    transform = osr.CoordinateTransformation(
        input_ds.GetSpatialRef(), target_srs)
    # print('err1')
    # Reproject the input dataset
    options = ['COMPRESS=' + compression, 'PREDICTOR=2']
    reprojected_ds = gdal.Warp(output_file_path, input_ds, dstSRS=target_projection,
                               options=[f'COMPRESS={compression}', 'PREDICTOR=2', f'ZLEVEL={zlevel}'], outputType=output_dtype)
    # print('err2')
    # Close the datasets
    input_ds = None
    reprojected_ds = None


def convert_geotransform(geotransform, src_projection, dst_projection):
    """
    Convert a geotransform from one projection to another.

    Parameters:
    - geotransform (tuple): Geotransform coefficients (originX, pixelWidth, rotationX, originY, rotationY, pixelHeight).
    - src_projection (str): Source projection in WKT format.
    - dst_projection (str): Destination projection in WKT format.

    Returns:
    - converted_geotransform (tuple): Converted geotransform coefficients in the destination projection.
    """
    # Create spatial reference objects for the source and destination projections
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_projection)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(dst_projection)

    # Create a coordinate transformation object from source to destination
    transform = osr.CoordinateTransformation(src_srs, dst_srs)

    # Transform the geotransform coefficients
    x, y, z = transform.TransformPoint(geotransform[0], geotransform[3])

    # Create the converted geotransform
    converted_geotransform = (
        x, geotransform[1], geotransform[2], y, geotransform[4], geotransform[5])

    return converted_geotransform


def crop_and_process_geoTiff(geoTiff, process_func, max_grid_shape=None, dtype = 'uint16'):
    """
    Split a GeoTIFF into grids, process each grid, and write the processed grids to an output GeoTIFF.

    Parameters:
    - dataset (gdal.Dataset): Input GeoTIFF dataset.
    - max_grid_shape (tuple): Maximum shape of each grid (rows, columns).
    - output_tiff (str): Output GeoTIFF file path.
    Raises:
    - ValueError: If the max_grid_shape is invalid or exceeds the size of the GeoTIFF.
    """
    if max_grid_shape is None:
        max_grid_shape = geoTiff.RasterYSize, geoTiff.RasterXSize
    if not isinstance(max_grid_shape, tuple) or len(max_grid_shape) != 2 or \
       max_grid_shape[0] <= 0 or max_grid_shape[1] <= 0:
        raise ValueError(
            "Invalid max_grid_shape. It should be a tuple of two positive integers.")

    if max_grid_shape[0] > geoTiff.RasterYSize or max_grid_shape[1] > geoTiff.RasterXSize:
        raise ValueError("max_grid_shape exceeds the size of the GeoTIFF.")

    # Iterate over each grid individually
    for i, grid in enumerate(crop_geoTiff_into_grids(geoTiff, max_grid_shape, True, dtype)):
        # Process each grid (Replace with your processing code)
        np_gt_part, extent, image_coord, progress = grid
        processed_grid = process_func(np_gt_part, extent, image_coord, progress)


def clip_geoTiff(input_layer, mask_layer, output_path, preprocess_func=None,  transform_to_mask_layer=False, output_dtype=None, verbose=True):
    """
    Clip a GeoTIFF image to the extent of a mask layer.

    Parameters:
    - input_layer (gdal.Dataset): Input GeoTIFF dataset to be clipped.
    - mask_layer (gdal.Dataset): Mask layer GeoTIFF dataset defining the clipping extent.
    - output_path (str): Output path for the clipped GeoTIFF file.
    - preprocess_func (function, optional): A function to preprocess the clipped image before saving.
    - transform_to_mask_layer (bool, optional): Flag indicating whether to transform the input layer to the extent and resolution of the mask layer. Defaults to False.
    - output_dtype (int, optional): Output data type for the clipped GeoTIFF.
    - verbose (bool, optional): If True, print messages during processing.

    Returns:
    - dest_gt (gdal.Dataset): Clipped GeoTIFF dataset.
    """
    intersection_extent = calculate_geoTiff_intersection_extent(
        input_layer, mask_layer)
    # print(intersection_extent)
    if intersection_extent is None:
        if verbose:
            print('Layers to not overlap.')
        return None
    ie_left, ie_right, ie_top, ie_bottom = intersection_extent
    input_l_image_coord = get_geoTiff_image_coordinates(
        input_layer, intersection_extent)
    left, right, top, bottom = input_l_image_coord
    np_input_l = crop_geoTiff(input_layer, left, top, right, bottom)
    if preprocess_func is not None:
       np_input_l = preprocess_func(np_input_l)
    if output_dtype is None:
       output_dtype = get_gdal_datatype_from_numpy_datatype(
           str(np_input_l.dtype))
    if transform_to_mask_layer:
        xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i = mask_layer.GetGeoTransform()
        xmin_i, ymin_i = ie_left, ie_top
        new_geoTransform = xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i
        # Resize the input layer to match the resolution of the mask layer
        mask_l_image_coord = get_geoTiff_image_coordinates(
            mask_layer, intersection_extent)
        left, right, top, bottom = mask_l_image_coord
        width, height = int(right - left), int(bottom - top)
        np_input_l = resize_np_image(np_input_l, (width, height))
        dest_gt = create_new_geoTiff(output_path, (width, height), np_input_l.shape[2],
                                     mask_layer.GetProjection(), new_geoTransform, output_dtype)

    else:
        xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i = input_layer.GetGeoTransform()
        xmin_i, ymin_i = ie_left, ie_top
        new_geoTransform = xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i
        width, height = int(right - left), int(bottom - top)
        dest_gt = create_new_geoTiff(output_path, (width, height),  np_input_l.shape[2],
                                     input_layer.GetProjection(), new_geoTransform, output_dtype)
    write_geoTiff_bands(dest_gt, np_input_l, (0, 0))
    return dest_gt


def clip_geoTiff_to_layer(input_layer, mask_layer, output_layer, preprocess_func=None,  transform_to_mask_layer=False, output_gdal_dtype=None, verbose=True):
    """
    Clip a GeoTIFF image to the extent of a mask layer.

    Parameters:
    - input_layer (gdal.Dataset): Input GeoTIFF dataset to be clipped.
    - mask_layer (gdal.Dataset): Mask layer GeoTIFF dataset defining the clipping extent.
    - output_path (str): Output path for the clipped GeoTIFF file.
    - transform_to_mask_layer (bool, optional): Flag indicating whether 
    to transform the input layer to the extent and resolution of the mask layer. Defaults to False.

    Returns:
    - dest_gt (gdal.Dataset): Clipped GeoTIFF dataset.

    """
    intersection_extent = calculate_geoTiff_intersection_extent(
        input_layer, mask_layer)
    # print(intersection_extent)
    if intersection_extent is None:
        if verbose:
           print('Layers to not overlap.')
        return None
    ie_left, ie_right, ie_top, ie_bottom = intersection_extent
    input_l_image_coord = get_geoTiff_image_coordinates(
        input_layer, intersection_extent)
    left, right, top, bottom = input_l_image_coord
    np_input_l = crop_geoTiff(input_layer, left, top, right, bottom)
    if preprocess_func is not None:
       np_input_l = preprocess_func(np_input_l)
    if output_gdal_dtype is None:
       output_gdal_dtype = get_gdal_datatype_from_numpy_datatype(
           str(np_input_l.dtype))
    if transform_to_mask_layer:
        xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i = mask_layer.GetGeoTransform()
        xmin_i, ymin_i = ie_left, ie_top
        new_geoTransform = xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i
        # Resize the input layer to match the resolution of the mask layer
        mask_l_image_coord = get_geoTiff_image_coordinates(
            mask_layer, intersection_extent)
        left, right, top, bottom = mask_l_image_coord
        width, height = int(right - left), int(bottom - top)
        np_input_l = resize_np_image(np_input_l, (width, height))
    else:
        xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i = input_layer.GetGeoTransform()
        xmin_i, ymin_i = ie_left, ie_top
        new_geoTransform = xmin_i, xres_i, xskew_i, ymin_i, yskew_r, yres_i
        # Resize the input layer to match the resolution of the mask layer
        input_l_image_coord = get_geoTiff_image_coordinates(
            input_layer, intersection_extent)
        left, right, top, bottom = input_l_image_coord
        width, height = int(right - left), int(bottom - top)
        np_input_l = resize_np_image(np_input_l, (width, height))

    if isinstance(output_layer, str):
        # print(np_input_l.shape)
        output_gt = create_new_geoTiff(output_layer, (width, height), np_input_l.shape[2],input_layer.GetProjection(), 
                                input_layer.GetGeoTransform(), output_gdal_dtype)
    else:
        output_gt = output_layer
    write_geoTiff_bands(output_gt, np_input_l, (left, top))
