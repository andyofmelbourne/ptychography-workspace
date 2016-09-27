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

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def make_prop(Fresnel, shape):
    if Fresnel is not False :
        i     = np.fft.fftfreq(shape[0], 1/float(shape[0]))
        j     = np.fft.fftfreq(shape[1], 1/float(shape[1]))
        i, j  = np.meshgrid(i, j, indexing='ij')
        
        # apply phase
        exps = np.exp(1.0J * np.pi * (i**2 * Fresnel / shape[0]**2 + \
                                      j**2 * Fresnel / shape[1]**2))
        def prop(x):
            out = np.fft.ifftn(np.fft.ifftshift(x, axes=(-2,-1)), axes = (-2, -1)) * exps.conj() 
            out = np.fft.fftn(out, axes = (-2, -1))
            return out
        
        def iprop(x):
            out = np.fft.ifftn(x, axes = (-2, -1)) * exps
            out = np.fft.fftn(out, axes = (-2, -1))
            return np.fft.ifftshift(out, axes=(-2,-1))

        #P = iprop(np.fft.fftn(np.fft.ifftshift(P)))
    else :
        def prop(x):
            out = np.fft.ifftshift(x, axes = (-2,-1))
            out = np.fft.fftn(out, axes = (-2, -1))
            return out
        
        def iprop(x):
            out = np.fft.ifftn(x, axes = (-2, -1))
            return np.fft.fftshift(out, axes = (-2, -1))
    return prop, iprop, exps

if __name__ == '__main__':
    #args, params = parse_cmdline_args()
    #f = h5py.File(args.filename)
    f = h5py.File('hdf5/MLL_520/MLL_520_cropped_binned.pty')
    g = h5py.File('hdf5/MLL_520/MLL_520_cropped_binned_output.pty')
    P_in  = g['input/P'][()]
    P_out = g['output/P'][()]

    # get the Fresnel number 
    ########################
    defocus = 0.00065
    lamb = f['metadata/wavelength'][()]
    z    = f['metadata/detector_distance'][()]
    du   = f['metadata/fs_pixel_size'][()]
    dq   = du / (lamb * z)
    Fresnel = 1 / (dq**2 * lamb * defocus)

    # propagate
    ###########
    prop, iprop, exps = make_prop(Fresnel, P_in.shape)

    """
    # from Stockmar2013-SciRep-near-field-ptychog
    M  = f['/metadata/detector_distance'].value / defocus
    dx = f['metadata/fs_pixel_size'][()] / M
    
    shape = P_in.shape
    i     = np.fft.fftfreq(shape[0], d=dx)
    j     = np.fft.fftfreq(shape[1], d=dx)
    i, j  = np.meshgrid(i, j, indexing='ij')

    q2 = (i**2+j**2)
    exp2 = np.exp(-2.0J*np.pi*defocus/(lamb * np.sqrt(1. - lamb**2 * q2)))
    """
    
