from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np

def loadmatdata(mat_fname):
    mat_contents = sio.loadmat(mat_fname)
    all_var_names = mat_contents.keys()
    var_names = [vn for vn in all_var_names if not ('__' in vn)]
    data = mat_contents[var_names[0]]
    return (data, var_names[0])

def writehdr(I, fname):
    hdr_lines = ['ENVI',
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

    with open(hdr_fname, 'w') as f:
        f.write('\n'.join(hdr_lines))

def writebip(I, fname):
    I.flatten().tofile(fname)
    writehdr(I, fname)

def convert_mat_to_bip(mat_filename,f_directory=None):
    mat_fname = mat_filename

    if f_directory!=None:
        fpath = f_directory
    else:
        fpath = './'

    I, varname = loadmatdata(fpath + mat_fname)
    print(varname)
    print(I.shape)

    bip_fname = f'{varname}.bip'
    writebip(I, fpath + bip_fname)
    return bip_fname
