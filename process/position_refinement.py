
import sys, os

import numpy as np
import h5py
import scipy.constants as sc
import time
from scipy import ndimage

import ConfigParser

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'process'))

import Ptychography.ptychography.era as era
from Ptychography import utils


def stitch_error_R(O, P, r, I, window=7):
    # make 7x7 forward estimates of I
    # may need to expand O to allow for edges...
    offsets = np.arange(-(window//2), window//2+1, 1)
    xs = r[1] + offsets
    ys = r[0] + offsets
    ys, xs = np.meshgrid(ys, xs, indexing='ij')
    
    R = np.zeros((window**2,3), dtype=r.dtype)
    R[:, 0] = ys.ravel()
    R[:, 1] = xs.ravel()

    Is = np.zeros((window**2,) + I.shape, dtype=I.dtype) 
    Is = era.make_exits(O, P, R, Is)

    errors = [np.sqrt(np.sum( (I - i)**2 )) for i in Is]
    i      = np.argmin(errors)
    return i, R[i], errors[i], errors[window**2//2]

def refine_positions(O, P, R, I, window=7):
    R_out = np.zeros_like(R)
    
    for i in range(len(R)):
        index, r, new_err, old_err = stitch_error_R(O, P, R[i], f['data'][i], window)
        R_out[i, 0] = r[0]
        R_out[i, 1] = r[1]
        era.update_progress(i / max(1.0, float(len(R)-1)), 'refining Rs:', index, old_err, new_err )
    return R_out 

if __name__ == '__main__':
    #f = h5py.File('hdf5/MLL_520/MLL_520_cropped_binned.pty')
    f = h5py.File(sys.argv[1])
    
    # stitch P
    ###########
    P = f['stitch/whitefield'][()].real
    
    # stitch O
    ##########
    O = f['stitch/O'][()].real
    
    # stitch R
    ##########
    R = f['stitch/R'][()].real
    R -= np.max(R, axis=0)

    # right now I_i ~ O(r-R_i) * P
    #errors, I_out, I_in = stitch_error_R(O, P, R[47], f['data'][47], 7)
    R_out = refine_positions(O, P, R, f['data'], window=7)

    # insert new positions in meters
    key = '/stitch/R'
    if key in f:
        del f[key]
    f[key] = R_out
    f.close()

    # get the Magnification
    M = f['/metadata/detector_distance'][()] / f['stitch/defocus'][()]
    
    # scale R into detector pixels
    R_out = R_out.astype(np.float)
    R_out[:, 0] *= f['/metadata/ss_pixel_size'].value / M
    R_out[:, 1] *= f['/metadata/fs_pixel_size'].value / M
    
    key = '/metadata/R_meters'
    if key in f:
        del f[key]
    f[key] = R_out
