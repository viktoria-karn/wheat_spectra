from skimage.measure import label, regionprops
from skimage.io import imsave
import numpy as np
import csv
import skimage
from scipy.signal import savgol_filter

def square_centr(image_in_array, image_out_path=None):
    label_img = label(image_in_array)

    for index in range(0, label_img.max()):
        props = regionprops(label_img)
        x = int(props[index].centroid[1])
        y = int(props[index].centroid[0])
        for i in range(y - 2, y + 3):
            for j in range(x - 2, x + 3):
                label_img[i][j] = -1
        label_img[label_img == props[index].label] = 0
        label_img[label_img == -1] = props[index].label
    if image_out_path != None:
        imsave(image_out_path, label_img)
    return label_img

def image_index(image_in_array, borders=None):
    label_img = label(image_in_array)
    if borders == None:
        borders = [0,0,0,0,0,0]
        borders[0] = 0
        borders[1] = 43
        borders[2] = 79
        borders[3] = 114
        borders[4] = 151
        borders[5] = 191
    for index in range(0, label_img.max()):
        props = regionprops(label_img)
        if borders[0] < props[index].centroid[1] < borders[1]:
            buf = props[index].label
            props[index].label = int((props[index].label - 1) / 5) * 5 + 1
            label_img[label_img == props[index].label] = -1
            label_img[label_img == buf] = props[index].label
            label_img[label_img == -1] = buf
        elif borders[1] < props[index].centroid[1] < borders[2]:
            buf = props[index].label
            props[index].label = int((props[index].label - 1) / 5) * 5 + 2
            label_img[label_img == props[index].label] = -1
            label_img[label_img == buf] = props[index].label
            label_img[label_img == -1] = buf
        elif borders[2] < props[index].centroid[1] < borders[3]:
            buf = props[index].label
            props[index].label = int((props[index].label - 1) / 5) * 5 + 3
            label_img[label_img == props[index].label] = -1
            label_img[label_img == buf] = props[index].label
            label_img[label_img == -1] = buf
        elif borders[3] < props[index].centroid[1] < borders[4]:
            buf = props[index].label
            props[index].label = int((props[index].label - 1) / 5) * 5 + 4
            label_img[label_img == props[index].label] = -1
            label_img[label_img == buf] = props[index].label
            label_img[label_img == -1] = buf
        elif borders[4] < props[index].centroid[1] < borders[5]:
            buf = props[index].label
            props[index].label = int((props[index].label - 1) / 5) * 5 + 5
            label_img[label_img == props[index].label] = -1
            label_img[label_img == buf] = props[index].label
            label_img[label_img == -1] = buf
    return label_img

def extract_gray_matr_spectrs(image_array,image_out_path):
    r_min = 139
    r_max = 148

    g_min = 141
    g_max = 157

    b_min = 84
    b_max = 99
    
    pixel_spectr_coord = []
    image_out_array = image_array

    with open("extract_spectrs.csv","w") as f:
        writer = csv.writer(f)
        for i in range(len(image_array)):
            for j in range(len(image_array[i])):
                if r_min <= int(image_array[i][j][0]) <= r_max and g_min <= int(image_array[i][j][1]) <= g_max and b_min <= int(
                        image_array[i][j][2]) <= b_max:
                    image_out_array[i][j] = 255
                    pixel_spectr_coord.append([i, j])
                    writer.writerow([i, j])
    if image_out_path!=None:
        skimage.io.imsave(image_out_path, image_out_array)

    return pixel_spectr_coord

def matr_select_spectrs(label_img, y_three_dimensional,filename,pixel_spectr_coord):
    i_offset = 144
    j_offset = 88
    grains = []
    spectr_not_aver = []

    for row in pixel_spectr_coord:
        i = int(row[0])
        j = int(row[1])
        if int(label_img[i][j]) != 0:
            row_1 = list(y_three_dimensional[i+i_offset][j+j_offset])
            row_1 = np.array(row_1)
            #snv
            row_1 = (row_1 - np.mean(row_1)) / np.std(row_1)
            row_1 = list(row_1)
            row = [label_img[i][j], i, j, row_1]
            grains.append(row)

            row = list(y_three_dimensional[i+i_offset][j+j_offset])
            row = np.array(row)
            # snv
            row = (row - np.mean(row)) / np.std(row)
            row = list(row)
            row.insert(0, filename)
            row.insert(1, label_img[i][j])
            row.insert(2, i)
            row.insert(3, j)
            spectr_not_aver.append(row)

    grains.sort()
    spectr_not_aver.sort()

    grains_mean = []
    matr_x = []
    spectr_matr = []
    number_label_img = 1

    for i in range(len(grains)):
        if number_label_img == grains[i][0]:
            spectr_matr.append(list(grains[i][3]))
        if number_label_img != grains[i][0] or i == len(grains) - 1:
            spectr_matr_transpose = [[spectr_matr[j][i] for j in range(len(spectr_matr))] for i in
                                     range(len(spectr_matr[0]))]

            spectr_matr = []
            spectr_mean = []
            for row in spectr_matr_transpose:
                spectr_mean.append(np.mean(row))
            row = spectr_mean
            row.insert(0, filename)
            row.insert(1, number_label_img)
            grains_mean.append(row)
            matr_x.append(spectr_mean)
            number_label_img += 1
            spectr_matr.append(list(grains[i][3]))

    return spectr_not_aver,grains_mean

def matr_average_spectrs(label_img, y_three_dimensional,filename):
    i_offset = 144
    j_offset = 88
    grains = []
    spectr_not_aver = []

    for i in range(len(label_img)):
        for j in range(len(label_img[i])):
            if label_img[i][j] != 0:
                row_1 = list(y_three_dimensional[i+i_offset][j+j_offset])

                row_1 = savgol_filter(row_1, 13, 3)
                row_1 = np.array(row_1)
                row_1 = (row_1 - np.mean(row_1)) / np.std(row_1)
                row_1 = list(row_1)
                row = [label_img[i][j], i, j, row_1]
                grains.append(row)

                row = list(y_three_dimensional[i+i_offset][j+j_offset])
                row = np.array(row)
                row = (row - np.mean(row)) / np.std(row)
                row = list(row)
                row.insert(0, filename)
                row.insert(1,label_img[i][j])
                row.insert(2, i)
                row.insert(3, j)
                spectr_not_aver.append(row)

    grains.sort()
    spectr_not_aver.sort()

    grains_mean = []
    matr_x = []
    spectr_matr = []
    number_label_img = 1

    for i in range(len(grains)):
        if number_label_img == grains[i][0]:
            spectr_matr.append(list(grains[i][3]))
        if number_label_img != grains[i][0] or i == len(grains) - 1:
            spectr_matr_transpose = [[spectr_matr[j][i] for j in range(len(spectr_matr))] for i in
                                     range(len(spectr_matr[0]))]

            spectr_matr = []
            spectr_mean = []
            for row in spectr_matr_transpose:
                spectr_mean.append(np.mean(row))

            row = spectr_mean
            row.insert(0, filename)
            row.insert(1, number_label_img)
            grains_mean.append(row)
            matr_x.append(spectr_mean)
            number_label_img += 1
            spectr_matr.append(list(grains[i][3]))
    return matr_x,spectr_not_aver

def add_protein(filename, filename_data=None, numbers_data=None):
    if numbers_data is None:
        number_row = 4
        number_column = 5
        number_protein = 9
    if filename_data is None:
        filename_data = "BIG DATA.csv"
    flag_first_str = 0
    matr_y_with_number = []
    with open(filename_data) as f:
        for row in csv.reader(f):
            if flag_first_str == 0:
                flag_first_str = 1
                continue
            if str(row[2]) == filename:
                number_object = (int(row[number_row]) - 1) * 5 + int(row[number_column])
                protein = np.float64(row[number_protein])
                matr_y_with_number.append([number_object, protein])

    matr_y_with_number.sort()
    return matr_y_with_number

def add_sort_name(filename, filename_data=None, numbers_data=None):
    if numbers_data is None:
        number_row = 4
        number_column = 5
        number_protein = 9
        number_sort = 7
        number_image = 2
    if filename_data is None:
        filename_data = "BIG DATA.csv"
    flag_first_str = 0
    matr_y_with_data = []
    with open(filename_data) as f:
        for row in csv.reader(f):
            if flag_first_str == 0:
                flag_first_str = 1
                continue
            if str(row[2]) == filename:
                number_object = (int(row[number_row]) - 1) * 5 + int(row[number_column])
                protein = np.float64(row[number_protein])
                matr_y_with_data.append([number_object, protein,row[number_sort],row[number_image]])

    matr_y_with_data.sort()
    return matr_y_with_data

def add_protein_edit_x(filename, matr_x, filename_data=None, numbers_data=None):
    if numbers_data is None:
        number_row = 4
        number_column = 5
        number_protein = 9
    if filename_data is None:
        filename_data = "BIG DATA.csv"
    flag_first_str = 0
    matr_y_with_number = []
    matr_x_new = []
    with open(filename_data) as f:
        for row in csv.reader(f):
            if flag_first_str == 0:
                flag_first_str = 1
                continue
            if str(row[2]) == filename:
                number_object = (int(row[number_row]) - 1) * 5 + int(row[number_column])
                protein = np.float64(row[number_protein])
                matr_y_with_number.append([number_object, protein])
                matr_x.append()

    matr_y_with_number.sort()
    return matr_y_with_number

def matr_y_without_number(matr_y_with_number):
    matr_y = []
    for row in matr_y_with_number:
        matr_y.append([row[1:]])
    return matr_y

def create_matr_fit(matr_x, matr_y, filename_matr_fit_x=None, filename_matr_fit_y=None):
    if filename_matr_fit_x is None:
        filename_matr_fit_x = "matr_fit_x.csv"
    if filename_matr_fit_y is None:
        filename_matr_fit_y = "matr_fit_y.csv"
    with open(filename_matr_fit_x, "w") as f:
        writer = csv.writer(f)
        for row in matr_x:
            writer.writerow(row)

    with open("matr_fit_y.csv", "w") as f:
        writer = csv.writer(f)
        for row in matr_y:
            writer.writerow(row)

def get_matr_fit_from_files(filename_matr_fit_x=None, filename_matr_fit_y=None):
    if filename_matr_fit_x is None:
        filename_matr_fit_x = "matr_fit_x.csv"
    if filename_matr_fit_y is None:
        filename_matr_fit_y = "matr_fit_y.csv"

    matr_fit_x = []
    matr_fit_y = []
    with open(filename_matr_fit_x) as f:
        for row in csv.reader(f):
            matr_fit_x.append(row)

    with open(filename_matr_fit_y) as f:
        for row in csv.reader(f):
            matr_fit_y.append(row)
    return matr_fit_x, matr_fit_y
