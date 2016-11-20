import scipy.constants as sc
import h5py
import numpy as np

import sys, os

import time
import ConfigParser

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'process'))

from Ptychography import utils
import utils as utils2
import optics 
from numpy.polynomial import polynomial as P
from numpy.polynomial import legendre as L

# I need a function like approx2 (maybe just use indexing)
# I need to fit polynomials to the phase gradients (not the phase)
# I need a function to invert the index maps

class Cpu_stitcher():

    def __init__(self, data, mask, W, R, O, X_ij): 
	dtype     = np.float64
	self.IW   = mask * data * W
        self.Od   = np.zeros_like(data, dtype=dtype)
	self.mask = mask
	self.WW   = mask * W**2
	self.R    = np.rint(R).astype(np.int)
	
	# add room for half a data frame
        self.R[:, 0] -= data.shape[1]//2
        self.R[:, 1] -= data.shape[2]//2
	
	if O is None :
	    self.O  = O
	else :
	    self.O  = O
	
        # the regular pixel values
        self.i, self.j = np.indices(self.IW.shape[1 :])

	if X_ij is not None :
	    self.X_ij = np.rint(X_ij).astype(np.int)
	else :
	    self.X_ij = np.zeros_like([self.i, self.j])
         
        # make the object grid
        Oshape = (int(round(self.i.max() + np.max(np.abs(self.R[:, 0])) + data.shape[1]//2)), \
                  int(round(self.j.max() + np.max(np.abs(self.R[:, 1])) + data.shape[2]//2)))
        
        if O is None :
            self.O = np.zeros(Oshape, dtype=dtype)
        else :
	    self.O = O

	self.WWmap = np.zeros_like(self.O)
    
    def forward_map(self, X_ij):
	for k in range(self.IW.shape[0]):
	    self.Od[k] =  self.O[self.i + X_ij[0] - self.R[k][0], self.j + X_ij[1] - self.R[k][1]] 
	return self.Od
    
    def inverse_map(self, X_ij):
	self.O.fill(0)
	self.WWmap.fill(0)
	for k in range(self.IW.shape[0]):
	    self.O[    self.i + X_ij[0] - self.R[k][0], self.j + X_ij[1] - self.R[k][1]] += self.IW[k]
	    self.WWmap[self.i + X_ij[0] - self.R[k][0], self.j + X_ij[1] - self.R[k][1]] += self.WW
	self.O /= (self.WWmap + 1.0e-5)
	return self.O
    
    def calc_error(self, X_ij):
        self.O       = self.inverse_map(X_ij)
        self.Od      = self.forward_map(X_ij)
        
        # sum |sqrt(I) - sqrt(I_forward)|^2
        self.error_map  = (np.sqrt(self.IW) - ap.sqrt(self.Od * self.WW))**2 / \
                          (1.0e-5+self.W)
        error = ap.sum( self.error_map )
        return error

    

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

    # delta_ij
    # -------------------
    if params['gpu_stitch']['delta_ij'] is not None and 'pixel_shifts_ss_gpu_stitch' in f[params['gpu_stitch']['h5_group']].keys():
        delta_ij    = np.zeros((2,) + f['/entry_1/data_1/data'].shape[1:], dtype=np.float)
        delta_ij[0] = f[params['gpu_stitch']['h5_group']]['pixel_shifts_ss_gpu_stitch']
        delta_ij[1] = f[params['gpu_stitch']['h5_group']]['pixel_shifts_fs_gpu_stitch']
        delta_ij    = delta_ij[:, ROI[0]:ROI[1], ROI[2]:ROI[3]]
    else :
        delta_ij    = None

    cpu_stitcher = Cpu_stitcher(data, mask, W, R, None, delta_ij)
    O = cpu_stitcher.inverse_map(cpu_stitcher.X_ij)

