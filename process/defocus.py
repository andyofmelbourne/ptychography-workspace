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

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='calculate the defocused probe')
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
        args.config = os.path.join(os.path.split(args.filename)[0], 'defocus.ini')
        if not os.path.exists(args.config):
            args.config = '../process/defocus.ini'
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = utils.parse_parameters(config)
    
    return args, params

def get_sample_plane_probe(p, phase, E, z, du, df):
    import scipy.constants as sc
    lamb = sc.h * sc.c / E
    
    # zero pad
    P2 = np.zeros( (2*p.shape[0], 2*p.shape[1]), dtype=p.dtype)
    P2[:p.shape[0], :p.shape[1]] = p
    P2 = np.roll(P2, p.shape[0]//2, 0)
    P2 = np.roll(P2, p.shape[1]//2, 1)
    
    dq = du / (lamb * z)
    
    i = np.fft.fftfreq(P2.shape[0], 1/float(P2.shape[0])) * dq[0]
    j = np.fft.fftfreq(P2.shape[1], 1/float(P2.shape[1])) * dq[1]
    i, j = np.meshgrid(i, j, indexing='ij')
    
    exp = np.exp(-1J * np.pi * lamb * df * (i**2 + j**2))

    i = np.fft.fftfreq(phase.shape[0], 1/float(phase.shape[0])) * dq[0]
    j = np.fft.fftfreq(phase.shape[1], 1/float(phase.shape[1])) * dq[1]
    i, j = np.meshgrid(i, j, indexing='ij')
    phase +=  np.fft.fftshift(- np.pi * lamb * df * (i**2 + j**2))

    P3 = np.fft.ifftshift(P2) * exp
    P3 = np.fft.fftshift(np.fft.ifftn(P3))
    return P3, phase

if __name__ == '__main__':
    args, params = parse_cmdline_args()
    
    f = h5py.File(args.filename)
    
    ################################
    # Get the inputs
    # frames, df, R, O, W, ROI, mask
    ################################
    group = params['defocus']['h5_group']
    
    # ROI
    # ------------------
    if params['defocus']['roi'] is not None :
        ROI = params['defocus']['roi']
    else :
        ROI = [0, f['entry_1/data_1/data'].shape[1], 0, f['entry_1/data_1/data'].shape[2]]
    
    # df
    # ------------------
    # get the sample to detector distance
    if params['defocus']['defocus'] is not None :
        df = params['defocus']['defocus']
    else :
        df = f['/entry_1/sample_3/geometry/translation'][0, 2]
    
    # W
    # ------------------
    # get the whitefield
    W = f[params['defocus']['whitefield']][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
    
    # phase
    # ------------------
    # get the phase
    if params['defocus']['phase'] is not None :
        phase = f[params['defocus']['phase']][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
    else :
        phase = np.zeros_like(W)

    # mask
    # ------------------
    # mask hot / dead pixels
    if params['defocus']['mask'] is None :
        bit_mask = f['entry_1/instrument_1/detector_1/mask'].value
        # hot (4) and dead (8) pixels
        mask     = ~np.bitwise_and(bit_mask, 4 + 8).astype(np.bool) 
    else :
        mask = f[params['defocus']['mask']].value
    mask     = mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]

    # geom
    # ------------------
    import scipy.constants as sc
    du = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
    z  = f['/entry_1/instrument_1/detector_1/distance'][()]
    E  = f['/entry_1/instrument_1/source_1/energy'][()]
    f.close()

    if params['defocus']['dfs'] is not None :
        dfs = np.linspace(  params['defocus']['dfs'][0],params['defocus']['dfs'][1],params['defocus']['dfs'][2])
    else :
        dfs = None
    
    if dfs is None :
        if rank == 0 :
             probe_df, phase_df = get_sample_plane_probe(np.sqrt(W) * np.exp(1.J * phase), phase, E, z, du, df)

    
    # write the result 
    ##################
    if params['defocus']['output_file'] is not None :
        g = h5py.File(params['defocus']['output_file'])
        outputdir = os.path.split(params['defocus']['output_file'])[0]
    else :
        g = h5py.File(args.filename)
        outputdir = os.path.split(args.filename)[0]
    
    group = params['defocus']['h5_group']
    if group not in g:
        print g.keys()
        g.create_group(group)

    key = params['defocus']['h5_group']+'/phase_df'
    if key in g :
        del g[key]
    g[key] = phase_df

    key = params['defocus']['h5_group']+'/probe_df'
    if key in g :
        del g[key]
    g[key] = probe_df
    
    g.close()
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print e

    
