"""
"""

import scipy.constants as sc
import h5py
import numpy as np
import afnumpy as ap
import arrayfire as af

import sys, os

import time
import ConfigParser

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'process'))

from Ptychography import utils
import utils as utils2

def stitch_with_distortions_gpu():
    pass

def unstitch_with_distortions_gpu():
    pass

def calculate_distortions_from_aberrations():
    pass



"""
fnam = '/asap3/petra3/gpfs/p11/2016/data/11002249/processed/cxi_files/MLL_391.cxi'
ROI  = [80, 426, 60, 450] 
df   = 200.0e-6

f = h5py.File('/asap3/petra3/gpfs/p11/2016/data/11002249/processed/cxi_files/MLL_391.cxi', 'r')

du = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
z  = f['/entry_1/instrument_1/detector_1/distance'][()]
E  = f['/entry_1/instrument_1/source_1/energy'][()]
wavelen = sc.h * sc.c / E

good_frames = list(f['/process_3/good_frames'][()])
b           = f['/entry_1/instrument_1/detector_1/basis_vectors'][good_frames]
R           = f['/entry_1/sample_3/geometry/translation'][good_frames]
data        = np.array([f['/entry_1/data_1/data'][fi][ROI[0]:ROI[1], ROI[2]:ROI[3]] for fi in good_frames])

whitefield  = f['/process_2/powder'][ROI[0]:ROI[1], ROI[2]:ROI[3]]
whitefield[whitefield==0] = 1.
whitefield  = whitefield.astype(np.float) / float(f['/entry_1/data_1/data'].shape[0])

# for now just divide the data by the whitefield
data = (data.astype(np.float) / whitefield.astype(np.float)).astype(np.float32)

# get the magnified sample-shifts 
# -------------------------------
# the x and y positions along the pixel directions
R_ss_fs = np.array([np.dot(b[i], R[i]) for i in range(len(R))])
R_ss_fs[:, 0] /= du[0]
R_ss_fs[:, 1] /= du[1]

# I want the x, y coordinates in scaled pixel units
# divide R by the scaled pixel size
R_ss_fs /= (df / z) * du

# offset the sample shifts so they start at zero
R_ss_fs[:, 0] -= np.max(R_ss_fs[:, 0])
R_ss_fs[:, 1] -= np.max(R_ss_fs[:, 1])

# get the coordinates of each pixel in each frame in pixel units
# --------------------------------------------------------------
# the regular pixel values
i, j = np.indices(data.shape[1 :])

# pixel offsets due to the phase gradients
dfs, dss = 0, 0

# make the object grid
Oss =  i.max() + np.max(dss) + np.max(np.abs(R_ss_fs[:, 0]))
Ofs =  j.max() + np.max(dfs) + np.max(np.abs(R_ss_fs[:, 1]))

O = np.zeros((int(round(Oss)), int(round(Ofs))), dtype=np.float32)
Oss, Ofs = np.indices(O.shape)

get_i_k = lambda k : Oss + dss + R_ss_fs[k, 0]
get_j_k = lambda k : Ofs + dfs + R_ss_fs[k, 1]

# now stitch send stuff to the gpu 
data_g    = ap.array(data.astype(np.float))
mask_g    = ap.array(np.ones_like(data[0]).astype(np.float))
Oss_g     = ap.array(Oss.astype(np.float).ravel())
Ofs_g     = ap.array(Ofs.astype(np.float).ravel())
R_ss_fs_g = ap.array(R_ss_fs.astype(np.float))
O_g       = ap.array(O.astype(np.float))

get_i_k_g = lambda k : Oss_g + dss + R_ss_fs_g[k, 0]
get_j_k_g = lambda k : Ofs_g + dfs + R_ss_fs_g[k, 1]

b    = None
norm = None
ds = []
for k in range(len(good_frames)):
    if b is None :
        b = af.approx2( data_g[k].d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        norm = af.approx2( mask_g.d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        #ds.append(np.array(b).reshape(O.shape))
    else :
        b += af.approx2( data_g[k].d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        norm += af.approx2( mask_g.d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        #ds.append(np.array(b).reshape(O.shape))
"""
def OP_sup(data, R, W, O=None, mask=None, O_dx=None, iters=1):
    # get the coordinates of each pixel in each frame in pixel units
    # --------------------------------------------------------------
    if mask is None :
        mask = 1
    
    # the regular pixel values
    i, j = np.indices(data.shape[1 :])
     
    # pixel offsets due to the phase gradients
    dfs, dss = 0, 0
    
    # make the object grid
    Oss =  i.max() + np.max(dss) + np.max(np.abs(R[:, 0]))
    Ofs =  j.max() + np.max(dfs) + np.max(np.abs(R[:, 1]))
    
    if O is None :
        O = np.zeros((int(round(Oss)), int(round(Ofs))), dtype=np.float32)
    Oss, Ofs = np.indices(O.shape)
    
    if O_dx is not None :
        Oss = Oss.astype(np.float) * O_dx[0]
        Ofs = Ofs.astype(np.float) * O_dx[1]

    get_i_k = lambda k : Oss + dss + R[k, 0]
    get_j_k = lambda k : Ofs + dfs + R[k, 1]

    print('sending stuff to the gpu')
    # now stitch send stuff to the gpu 
    data_g    = ap.array((mask*data).astype(np.float))
    W_g       = ap.array((mask*W).astype(np.float))
    Oss_g     = ap.array(Oss.astype(np.float).ravel())
    Ofs_g     = ap.array(Ofs.astype(np.float).ravel())
    R_g       = ap.array(R.astype(np.float))
    O_g       = ap.array(O.astype(np.float).ravel())
    
    get_i_k_g = lambda k : Oss_g + dss + R[k, 0]
    get_j_k_g = lambda k : Ofs_g + dfs + R[k, 1]
    
    print('looping gpu in OP_sup:')
    norm = ap.zeros(O_g.shape, O_g.dtype)
    for k in range(data_g.shape[0]):
        print k
        O_g  += af.approx2( (W_g*data_g[k]).d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        norm += af.approx2( (W_g*W_g).d_array      , get_j_k_g(k).d_array, get_i_k_g(k).d_array)
    
    print('done with gpu OP_sup')
    norm  = np.array(norm)
    O_out = np.array(O_g)
    norm[norm==0] = 1
    return (O_out/norm).reshape(O.shape), np.array(W_g).reshape(W.shape)


def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='calculate a stitch of the projection images with phase gradients')
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
        args.config = os.path.join(os.path.split(args.filename)[0], 'gpu_stitch.ini')
        if not os.path.exists(args.config):
            args.config = '../process/gpu_stitch.ini'
    
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
    # Zernike polynomials
    ################################
    group = params['gpu_stitch']['h5_group']
    
    # ROI
    # ------------------
    if params['gpu_stitch']['roi'] is not None :
        ROI = params['gpu_stitch']['roi']
    else :
        ROI = [0, f['entry_1/data_1/data'].shape[0], 0, f['entry_1/data_1/data'].shape[1]]
    
    # frames
    # ------------------
    # get the frames to process
    if 'good_frames' in f[group] :
        good_frames = list(f[group]['good_frames'][()])
    else :
        good_frames = range(f['entry_1/data_1/data'].shape[0])
    
    data = np.array([f['/entry_1/data_1/data'][fi][ROI[0]:ROI[1], ROI[2]:ROI[3]] for fi in good_frames])
    
    # df
    # ------------------
    # get the sample to detector distance
    if params['gpu_stitch']['defocus'] is not None :
        df = params['gpu_stitch']['defocus']
    else :
        df = f['/entry_1/sample_3/geometry/translation'][0, 2]
    
    # R
    # ------------------
    # get the pixel shift coordinates along ss and fs
    R, du = utils2.get_Fresnel_pixel_shifts_cxi(f, good_frames, params['gpu_stitch']['defocus'], offset_to_zero=True)
    
    # allow for astigmatism
    if params['gpu_stitch']['defocus_fs'] is not None :
        R[:, 1] *= df / params['gpu_stitch']['defocus_fs']
    
    R = np.rint(R).astype(np.int)
    
    # O
    # ------------------
    # get the sample
    if params['gpu_stitch']['sample'] is not None :
        O = f[params['gpu_stitch']['sample']][()]
    elif params['gpu_stitch']['o_shape'] is not None :
        print(tuple(params['gpu_stitch']['o_shape']))
        O = np.zeros(tuple(params['gpu_stitch']['o_shape']), dtype=np.float64)
    else :
        O = None
    
    if params['gpu_stitch']['o_dx'] is not None :
        print('du:', du)
        dx = params['gpu_stitch']['o_dx'] / du
    else :
        dx = [1., 1.]
    print('dx:', dx)
    
    # W
    # ------------------
    # get the whitefield
    if params['gpu_stitch']['whitefield'] is not None :
        W = f[params['gpu_stitch']['whitefield']][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
    else :
        W = np.mean(data, axis=0)

    # mask
    # ------------------
    # mask hot / dead pixels
    if params['gpu_stitch']['mask'] is None :
        bit_mask = f['entry_1/instrument_1/detector_1/mask'].value
        # hot (4) and dead (8) pixels
        mask     = ~np.bitwise_and(bit_mask, 4 + 8).astype(np.bool) 
    else :
        mask = f[params['gpu_stitch']['mask']].value
    mask     = mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]

    # Zernike polynomials
    # -------------------
    # get the list of zernike polynomial coefficients 
    # if there are any
    if params['gpu_stitch']['zernike'] is not None :
        Z = f[params['gpu_stitch']['zernike']][()].astype(np.float)
    else :
        Z = np.zeros( (params['gpu_stitch']['orders'],), dtype=np.float)
    
    fit_grads = params['gpu_stitch']['fit_grads']
    orders    = params['gpu_stitch']['orders']
    
    #####################
    # Refine O and W
    #####################
    #O, P = OP_sup(data.astype(np.float), R, W, O, mask, O_dx = dx, iters=params['gpu_stitch']['iters'])
    O, P, Z_out, errors = fit_Zernike_grads(data.astype(np.float), R, W, O, mask, Z, \
                          fit_grads=fit_grads, orders=orders, O_dx = dx, iters=params['gpu_stitch']['iters'])
    
    if params['gpu_stitch']['normalise'] :
        a = np.mean(np.abs(O))
        P *= a
        O /= a
    
    W = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    W[ROI[0]:ROI[1], ROI[2]:ROI[3]] = P[:].real
    
    # write the result 
    ##################
    if params['gpu_stitch']['output_file'] is not None :
        g = h5py.File(params['gpu_stitch']['output_file'])
        outputdir = os.path.split(params['gpu_stitch']['output_file'])[0]
    else :
        g = f
        outputdir = os.path.split(args.filename)[0]
    
    key = params['gpu_stitch']['h5_group']+'/O_gpu_stitch'
    if key in g :
        del g[key]
    g[key] = O.astype(np.complex128)
    
    key = params['gpu_stitch']['h5_group']+'/whitefield_gpu_stitch'
    if key in g :
        del g[key]
    g[key] = W
    
    g.close()
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print e
