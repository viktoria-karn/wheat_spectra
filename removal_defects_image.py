import cv2
from scipy import misc, ndimage
import skimage
import numpy as np
from skimage.measure import label, regionprops, regionprops_table, find_contours
from skimage.color import label2rgb

def erosion(img, SE):
    imgErode = cv2.erode(img, SE, 1)
    return imgErode

def dilation(img, SE):
    imgDilate = cv2.dilate(img, SE, 1)
    return imgDilate

def open_close_image(image_in_array,image_out_path=None):
    # Define the structuring element using inbuilt CV2 function
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    arr = dilation(image_in_array, SE)
    arr = ndimage.binary_fill_holes(arr).astype(int)
    arr = arr.astype('uint8')
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    arr = erosion(arr, SE)

    # Erode the image
    AeB = erosion(arr, SE)
    # Dilate the eroded image. This gives opening operationAoB= dilation(AeB, SE)
    # dilate the opened image followed by ersoion. This will give closing of the opened image
    AoB = dilation(AeB, SE)
    AoBdB = dilation(AoB, SE)
    image_out_array = erosion(AoBdB, SE)
    image_out_array = image_out_array.astype(int)
    # Save the filtered image
    if image_out_path is not None:
        skimage.io.imsave(image_out_path, image_out_array)
    return image_out_array


def fill_holes(image_in_array,image_out_path=None):
    image_out_array = ndimage.binary_fill_holes(image_in_array).astype(int)
    if image_out_path!=None:
        skimage.io.imsave(image_out_path, image_out_array)
    return image_out_array


def removal_small_object(image_in_array,image_out_path=None):
    image_out_array = label(image_in_array)

    props = regionprops(image_out_array)
    all_props_area = 0
    for prop in props:
        all_props_area += prop.area

    mean_value = all_props_area/len(props)

    number_region = 0
    for region in regionprops(image_out_array):
        number_region += 1
        if region.area < mean_value / 4:
            image_out_array[image_out_array == number_region] = 0
        else:
            image_out_array[image_out_array == number_region] = 1

    skimage.io.imsave(image_out_path, image_out_array)
    return image_out_array
