import numpy as np
from scipy.signal import savgol_filter
from pylab import *
import time

def writehdr(I,fname):
  hdr_lines=['ENVI',
             'description = {}',
             f'samples = {I.shape[1]}',
             f'lines = {I.shape[0]}',
             f'bands = {I.shape[2]}',
             'header offset = 0',
             'file type = ENVI Standard',
             'data type = 5',
             'interleave = bip',
             'sensor type = Unknown',
             'byte order = 0']

  hdr_fname = f'{fname}.hdr'

  with open(hdr_fname,'w') as f:
    f.write('\n'.join(hdr_lines))

def read_bin_file(filename):
    f = open(filename, 'rb')
    values = np.fromfile(f, dtype="float")
    return values

def smoothing_savgol(fname,f_directory=None):
    if f_directory != None:
        fpath = f_directory
    else:
        fpath = './'

    with open(fpath+fname+".bip.hdr") as f:
        rows = f.readlines()
        samples = int(rows[2].split(" = ")[1])
        lines = int(rows[3].split(" = ")[1])
        bands = int(rows[4].split(" = ")[1])

    y = read_bin_file(fpath+fname+".bip")
    y_three_dimensional = y.reshape(lines, samples, bands)

    time_begin = time.time()
    for i in range(len(y_three_dimensional)):
        for j in range(len(y_three_dimensional[i])):
            y_three_dimensional[i][j] = savgol_filter(y_three_dimensional[i][j], 13, 3)

    y_three_dimensional.flatten().tofile(fname+"_savgol.bip")
    writehdr(y_three_dimensional, fname+"_savgol.bip")
    time_end = time.time() - time_begin
    print("Затраченное время", time_end)
    return y_three_dimensional

def get_3d_array(fname,f_directory=None):
    if f_directory != None:
        fpath = f_directory
    else:
        fpath = './'

    with open(fpath+fname+".bip.hdr") as f:
        rows = f.readlines()
        samples = int(rows[2].split(" = ")[1])
        lines = int(rows[3].split(" = ")[1])
        bands = int(rows[4].split(" = ")[1])

    y = read_bin_file(fpath+fname+".bip")
    y_three_dimensional = y.reshape(lines, samples, bands)


    return y_three_dimensional
