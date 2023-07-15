import numpy as np
import skimage
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import wheat_spectras.convert as con

def read_bin_file(filename):
    f = open(filename, 'rb')
    values = np.fromfile(f, dtype="float")
    return values

def averaged_matrix(filename, fname_image_gray=None):
    path = "mat files/"
    filename_smoothing = filename
    with open(path + filename_smoothing + ".bip.hdr") as f:
        rows = f.readlines()
        samples = int(rows[2].split(" = ")[1])
        lines = int(rows[3].split(" = ")[1])
        bands = int(rows[4].split(" = ")[1])

    # переписываем данные в трехмерный массив
    y = read_bin_file(path + filename_smoothing + ".bip")
    y_three_dimensional = y.reshape(lines, samples, bands)
    # получение средней матрицы из трехмерного массива
    y_matr = np.mean(y_three_dimensional, 2)
    # выполняем нормировку
    max_matr = y_matr.max()
    y_matr = y_matr / max_matr
    # 255-белый цвет, 0 - черный
    y_matr = y_matr * 255
    # преобразуем к целому типу
    y_matr = np.uint8(y_matr)
    # изображение в серых тонах
    if fname_image_gray != None:
        skimage.io.imsave(fname_image_gray, y_matr)
    return y_matr, y_three_dimensional

def get_matrix_RMS(filename, fname_image_gray=None):
    path = "mat files/"
    filename_smoothing = filename
    with open(path + filename_smoothing + ".bip.hdr") as f:
        rows = f.readlines()
        samples = int(rows[2].split(" = ")[1])
        lines = int(rows[3].split(" = ")[1])
        bands = int(rows[4].split(" = ")[1])

    # переписываем данные в трехмерный массив
    y = read_bin_file(path + filename_smoothing + ".bip")
    y_three_dimensional = y.reshape(lines, samples, bands)

    n = y_three_dimensional.shape[2]
    rms = ((y_three_dimensional ** 2).sum(axis=2) / n) ** 0.5

    rms_norm = y_three_dimensional / rms[:, :, None]

    varname = "rms_norm"
    bip_fname = f'{varname}.bip'

    con.writebip(rms_norm, bip_fname)
    # выделение rgb-каналов
    # 141 - красный (номер позиции во всем спектре)
    # 70 - зеленый (номер позиции во всем спектре)
    # 0 - синий (номер позиции во всем спектре)
    rms_norm_r = rms_norm[:, :, 141]
    rms_norm_g = rms_norm[:, :, 70]
    rms_norm_b = rms_norm[:, :, 0]
    
    #работа с красным
    min_rms = rms_norm_r.min()
    max_rms = rms_norm_r.max()
    rms_norm_r = (rms_norm_r - min_rms) / (max_rms - min_rms)
    rms_norm_r *= 255
    # преобразуем к целому типу
    rms_norm_r = np.rint(rms_norm_r)
    
    #работа с зеленым
    min_rms = rms_norm_g.min()
    max_rms = rms_norm_g.max()
    rms_norm_g = (rms_norm_g - min_rms) / (max_rms - min_rms)
    rms_norm_g *= 255
    # преобразуем к целому типу
    rms_norm_g = np.rint(rms_norm_g)
    
    #работа с синим
    min_rms = rms_norm_b.min()
    max_rms = rms_norm_b.max()
    rms_norm_b = (rms_norm_b - min_rms) / (max_rms - min_rms)
    rms_norm_b *= 255
    # преобразуем к целому типу
    rms_norm_b = np.rint(rms_norm_b)

    rms_norm_r = rms_norm_r[:, :, None]
    rms_norm_g = rms_norm_g[:, :, None]
    rms_norm_b = rms_norm_b[:, :, None]

    rms_all = np.concatenate([rms_norm_r, rms_norm_g, rms_norm_b], axis=2)

    if fname_image_gray != None:
        skimage.io.imsave(fname_image_gray, rms_all)
