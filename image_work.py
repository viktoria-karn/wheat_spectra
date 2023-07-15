from PIL import Image
import imageio
import os
import skimage
from skimage import filters
import matplotlib.pyplot as plt
import csv

def image_crop(image_in_path,cord=None,image_out_path=None):
    flag_remove_file=0
    # обрезка картинки в серых тонах
    picture = Image.open(image_in_path)
    if cord==None:
        cord = (88, 144, 280, 1857)
    picture = picture.crop(cord)
    if image_out_path==None:
        flag_remove_file=1
        image_out_path = "image_crop.png"
    picture = picture.save(image_out_path)
    # получение нового массива данных с обрезанной картинки
    image_out_array = imageio.imread(image_out_path)
    if flag_remove_file==1:
        os.remove(image_out_path)
    return image_out_array

def black_border(image_in_array,image_out_path=None):
    image_out_array = image_in_array
    for i in range(len(image_out_array)):
        # если первый элемент в строке черный, пропускаем
        if 0 <= image_out_array[i][0] <= 20:
            continue
        for j in range(len(image_out_array[i])):
            if 0 <= image_out_array[i][j] <= 20:
                break
            if 200 <= image_out_array[i][j] <= 255:
                image_out_array[i][j] = 0

    for i in range(len(image_out_array)):
        # если последний элемент в строке черный, пропускаем
        if 0 <= image_out_array[i][-1] <= 20:
            continue
        for j in range(len(image_out_array[i])-1,0,-1):
            if 0 <= image_out_array[i][j] <= 20:
                break
            if 150 <= image_out_array[i][j] <= 255:
                image_out_array[i][j] = 0

    with open("matr_x_gray.csv", "w") as f:
        writer = csv.writer(f)
        for row in image_out_array:
            writer.writerow(row)

    if image_out_path!=None:
        skimage.io.imsave(image_out_path, image_out_array)
    return image_out_array


def black_border_01(image_in_array,image_out_path=None):
    #1-белый
    #0-черный
    image_out_array = image_in_array
    for i in range(len(image_out_array)):
        # если первый элемент в строке черный, пропускаем
        if image_out_array[i][0] == 0:
            continue
        for j in range(len(image_out_array[i])):
            if image_out_array[i][j] == 0:
                break
            if image_out_array[i][j] ==1:
                image_out_array[i][j] = 0

    for i in range(len(image_out_array)):
        # если последний элемент в строке черный, пропускаем
        if image_out_array[i][-1] == 0:
            continue
        for j in range(len(image_out_array[i])-1,0,-1):
            if image_out_array[i][j] == 0:
                break
            if image_out_array[i][j] == 1:
                image_out_array[i][j] = 0

    if image_out_path!=None:
        skimage.io.imsave(image_out_path, image_out_array)
    return image_out_array


def image_mask(image_in_array,hist_path=None,image_out_path=None):
    thresh_min = filters.threshold_yen(image_in_array)
    #thresh_min = 20
    if hist_path!=None:
        plt.hist(image_in_array.ravel(), bins=256)
        plt.axvline(thresh_min, color='g')
        plt.savefig(hist_path)
    binary_min = image_in_array > thresh_min
    if image_out_path!=None:
        skimage.io.imsave(image_out_path, binary_min)
    image_out_array = skimage.img_as_ubyte(binary_min)
    image_out_array = image_out_array.astype('uint8')
    return image_out_array
