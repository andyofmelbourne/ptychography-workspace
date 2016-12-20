"""
take a *.pty file and calculate the stitched image

Needs to have the following structure:
$ h5ls -r hdf5/MLL_520/MLL_520.pty 
/                        Group
/R                       Dataset {119, 3}
/data                    Dataset {119, 516, 1556}
/good_frames             Dataset {119}
/whitefield              Dataset {516, 1556}
/mask                    Dataset {516, 1556}

/metadata                Group
/metadata/R_meters       Dataset {119, 3}
/metadata/defocus        Dataset {SCALAR}
/metadata/detector_distance Dataset {SCALAR}
/metadata/fs_pixel_size  Dataset {SCALAR}
/metadata/grid           Dataset {2}
/metadata/ss_pixel_size  Dataset {SCALAR}
/metadata/steps          Dataset {119}
/metadata/wavelength     Dataset {SCALAR}

the defocus is read from the configuration file

stitch[r] = \sum_i I[r - r_i] * W[r - r_i] / \sum W[r - r_i]^2

where W is the whitefield.
r is the detector pixel coordinate and r_i are the scaled sample coordinates.

Projection images of the sample are equivilant to an out of focus image of the sample:

plane waves
    |||  --> sample --> imaging plane --> lens -- > detector
            D = z1 . z2 / z1 + z2           M = z1 + z2 / z1
            D = defocus                     M = magnification

is the same as 
Focus
  . ---> sample --> detctor
     z1         z2

so the r_i values are then the scaled sample shift coordinates
in detector pixel units:

r_i = R_i * M / pixel_size 

where R_i are the sample shift coordinates in meters
"""
import sys, os

import numpy as np
import h5py
import scipy.constants as sc
import time
from scipy import ndimage

import ConfigParser

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

sys.path.append(root)
sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'process'))

import Ptychography.ptychography.era as era
from Ptychography import utils
import utils as utils2

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def make_P_heatmap(P, R, shape):
    P_heatmap = np.zeros(shape, dtype = P.real.dtype)
    #P_temp    = np.zeros(shape, dtype = P.real.dtype)
    #P_temp[:P.shape[0], :P.shape[1]] = (P.conj() * P).real
    P_temp = (P.conj() * P).real
    for r in R : 
        #P_heatmap += multiroll(P_temp, [-r[0], -r[1]]) 
        P_heatmap[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]] += P_temp
    return P_heatmap

def make_O_heatmap(O, R, shape):
    O_heatmap = np.zeros(O.shape, dtype = O.real.dtype)
    O_temp    = (O * O.conj()).real
    for r in R : 
        O_heatmap += era.multiroll(O_temp, [r[0], r[1]]) 
    return O_heatmap[:shape[0], :shape[1]]

def psup_P(exits, O, R, O_heatmap = None, alpha = 1.0e-10, MPI_dtype = MPI.DOUBLE, MPI_c_dtype = MPI.DOUBLE):
    PT = np.zeros(exits[0].shape, exits.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set O_heatmap = None when the object/coords has changed)
    if O_heatmap is None : 
        O_heatmapT = np.ascontiguousarray(make_O_heatmap(O, R, PT.shape))
        #O_heatmapT = era.make_O_heatmap(O, R, PT.shape) produces a non-contig. array for some reason
        O_heatmap  = np.empty_like(O_heatmapT)
        comm.Allreduce([O_heatmapT, MPI_dtype], \
                       [O_heatmap,  MPI_dtype], \
                       op=MPI.SUM)

    # Calculate numerator
    #--------------------
    Oc = O.conj()
    for r, exit in zip(R, exits):
        PT += exit * Oc[-r[0]:PT.shape[0]-r[0], -r[1]:PT.shape[1]-r[1]] 
         
    # divide
    #-------
    P = np.empty_like(PT)
    comm.Allreduce([PT, MPI_c_dtype], \
                   [P, MPI_c_dtype],   \
                    op=MPI.SUM)
    P  = P / (O_heatmap + alpha)
    
    return P, O_heatmap

def psup_O(exits, P, R, O_shape, P_heatmap = None, alpha = 1.0e-10, MPI_dtype = MPI.DOUBLE, MPI_c_dtype = MPI.DOUBLE, verbose = False, sample_blur = None):
    OT = np.zeros(O_shape, P.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set P_heatmap = None when the probe/coords has changed)
    if P_heatmap is None : 
        P_heatmapT = make_P_heatmap(P, R, O_shape)
        P_heatmap  = np.empty_like(P_heatmapT)
        #comm.Allreduce([P_heatmapT, MPI.__TypeDict__[P_heatmapT.dtype.char]], \
        #               [P_heatmap,  MPI.__TypeDict__[P_heatmap.dtype.char]], \
        #               op=MPI.SUM)
        comm.Allreduce([P_heatmapT, MPI_dtype], \
                       [P_heatmap,  MPI_dtype], \
                       op=MPI.SUM)

    # Calculate numerator
    #--------------------
    for r, exit in zip(R, exits):
        OT[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]] += exit * P.conj()
    
    # divide
    # here we need to do an all reduce
    #---------------------------------
    O = np.empty_like(OT)
    #comm.Allreduce([OT, MPI.__TypeDict__[OT.dtype.char]], \
    #               [O, MPI.__TypeDict__[O.dtype.char]],   \
    #                op=MPI.SUM)
    comm.Allreduce([OT, MPI_c_dtype], \
                   [O, MPI_c_dtype],  \
                    op=MPI.SUM)
    comm.Barrier()
    O  = O / (P_heatmap + alpha)

    if sample_blur is not None :
        import scipy.ndimage
        O.real = scipy.ndimage.gaussian_filter(O.real, sample_blur, mode='wrap')
        O.imag = scipy.ndimage.gaussian_filter(O.imag, sample_blur, mode='wrap')
    
    # set a maximum value for the amplitude of the object
    #O = np.clip(np.abs(O), 0.0, 2.0) * np.exp(1.0J * np.angle(O))
    return O, P_heatmap

def OP_sup(I, R, whitefield, O, mask, iters=4):
    if O is None :
        # find the smallest array that fits O
        # This is just U = M + R[:, 0].max() - R[:, 0].min()
        #              V = K + R[:, 1].max() - R[:, 1].min()
        shape = (I.shape[1] + R[:, 0].max() - R[:, 0].min(),\
                 I.shape[2] + R[:, 1].max() - R[:, 1].min())
        O = np.ones(shape, dtype = np.float64)
        print 'O.shape', O.shape

    # subtract an overall offset from R's
    R[:, 0] -= R[:, 0].max()
    R[:, 1] -= R[:, 1].max()
    
    #P = whitefield**2
    P = whitefield * mask
    I *= mask
    
    for i in range(iters):
        O0 = O.copy()
        O, P_heatmap = psup_O(I, P, R, O.shape, None)
        P, O_heatmap = psup_P(I, O, R)
        print i, np.sum( (O0 - O)**2 )
    return O, P


def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='calculate a basic stitch of the projection images')
    parser.add_argument('filename', type=str, \
                        help="file name of the *.pty file")
    parser.add_argument('-c', '--config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    # check that cxi file exists
    if not os.path.exists(args.filename):
        raise NameError('cxi file does not exist: ' + args.filename)
    
    # if config is non then read the default from the *.pty dir
    if args.config is None :
        args.config = os.path.join(os.path.split(args.filename)[0], 'basic_stitch.ini')
        if not os.path.exists(args.config):
            args.config = '../process/basic_stitch.ini'
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = utils.parse_parameters(config)
    
    return args, params


if __name__ == '__main__':
    args, params = parse_cmdline_args()
    f = h5py.File(args.filename)
    
    ################################
    # Get the inputs
    # frames, df, R, O, W, ROI, mask
    ################################
    group = params['stitch']['h5_group']
    
    # ROI
    # ------------------
    if params['stitch']['roi'] is not None :
        ROI = params['stitch']['roi']
    else :
        ROI = [0, f['entry_1/data_1/data'].shape[1], 0, f['entry_1/data_1/data'].shape[2]]
    
    # frames
    # ------------------
    # get the frames to process
    if 'process_3/good_frames' in f :
        good_frames = list(f['process_3/good_frames'][()])
    else :
        good_frames = range(f['entry_1/data_1/data'].shape[0])
    
    data = np.array([f['/entry_1/data_1/data'][fi][ROI[0]:ROI[1], ROI[2]:ROI[3]] for fi in good_frames])
    
    # df
    # ------------------
    # get the sample to detector distance
    if params['stitch']['defocus'] is not None :
        df = params['stitch']['defocus']
    else :
        df = f['/entry_1/sample_3/geometry/translation'][0, 2]
    
    # R
    # ------------------
    # get the pixel shift coordinates along ss and fs
    R, du = utils2.get_Fresnel_pixel_shifts_cxi(f, good_frames, params['stitch']['defocus'], offset_to_zero=True)
    
    # allow for astigmatism
    if params['stitch']['defocus_fs'] is not None :
        R[:, 1] *= df / params['stitch']['defocus_fs']
    
    R = np.rint(R).astype(np.int)
    
    # O
    # ------------------
    # get the sample
    if params['stitch']['sample'] is not None :
        O = f[params['stitch']['sample']][()]
    else :
        O = None
    
    # W
    # ------------------
    # get the whitefield
    if params['stitch']['whitefield'] is not None :
        W = f[params['stitch']['whitefield']][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
    else :
        W = np.mean(data, axis=0)

    # mask
    # ------------------
    # mask hot / dead pixels
    if params['stitch']['mask'] is None :
        bit_mask = f['entry_1/instrument_1/detector_1/mask'].value
        # hot (4) and dead (8) pixels
        mask     = ~np.bitwise_and(bit_mask, 4 + 8).astype(np.bool) 
    else :
        mask = f[params['stitch']['mask']].value
    mask     = mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]

    f.close()

    #####################
    # Refine O and W
    #####################
    O, P = OP_sup(data.astype(np.float), R, W, O, mask, iters=params['stitch']['iters'])

    if params['stitch']['normalise'] :
        a = np.mean(np.abs(O))
        P *= a
        O /= a

    f = h5py.File(args.filename)

    W = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    W[ROI[0]:ROI[1], ROI[2]:ROI[3]] = P[:].real
     
    # write the result 
    ##################
    if params['stitch']['output_file'] is not None :
        g = h5py.File(params['stitch']['output_file'])
        outputdir = os.path.split(params['stitch']['output_file'])[0]
    else :
        g = f
        outputdir = os.path.split(args.filename)[0]
    
    group = params['stitch']['h5_group']
    if group not in g:
        print g.keys()
        g.create_group(group)

    key = params['stitch']['h5_group']+'/O'
    if key in g :
        del g[key]
    g[key] = O.astype(np.complex128)
    
    key = params['stitch']['h5_group']+'/whitefield'
    if key in g :
        del g[key]
    g[key] = W
    
    """
    key = params['stitch']['h5_group']+'/R'
    if key in g :
        del g[key]
    print 'R shape:', R.shape
    g[key] = R

    key = params['stitch']['h5_group']+'/defocus'
    if key in g :
        del g[key]
    g[key] = params['stitch']['defocus']
    """
    
    g.close()
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print e
