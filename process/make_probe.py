"""
Get the white field and defocus values
then propagate the pupil to the sample plane 
then write it to the *.pty file
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

sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'process'))

import Ptychography.ptychography.era as era
from Ptychography import utils

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='calculate the sample plane probe function')
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
        args.config = os.path.join(os.path.split(args.filename)[0], 'make_probe.ini')
        if not os.path.exists(args.config):
            args.config = '../process/make_probe.ini'
    
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
    
    # get the Fresnel number 
    ########################
    defocus = params['make_probe']['defocus']
    lamb = f['metadata/wavelength'][()]
    z    = f['metadata/detector_distance'][()]
    du   = f['metadata/fs_pixel_size'][()]
    dq   = du / (lamb * z)
    if params['make_probe']['fresnel'] : 
        Fresnel = 1 / (dq**2 * lamb * defocus)
    else :
        Fresnel = False

    # pupil intensity
    #################
    I = f[params['make_probe']['use_whitefield']][()].astype(np.float)

    # Fill masked pixels
    ####################
    if params['make_probe']['fill_masked_pixels'] :
        pass

    # pupil phase
    #############
    if Fresnel is False :
        shape = I.shape
        # then add defocus to pupil phase
        # phase = exp( -i pi lambda z df q**2 ) 
        i     = np.fft.fftfreq(shape[0], 1/float(shape[0]))
        j     = np.fft.fftfreq(shape[1], 1/float(shape[1]))
        i, j  = np.meshgrid(i, j, indexing='ij')
        phase = -1.0J * np.pi * lamb * defocus * dq**2 * (i**2 + j**2)
    else :
        phase = np.zeros_like(I)
        
    P = np.sqrt(I) * np.fft.fftshift(np.exp(phase))
    
    # propagate
    ###########
    prop, iprop = era.make_prop(Fresnel, P.shape)
    
    P_sample = iprop(np.fft.ifftshift(P))
    
    # write the result 
    ##################
    if params['make_probe']['output_file'] is not None :
        g = h5py.File(params['make_probe']['output_file'])
        outputdir = os.path.split(params['make_probe']['output_file'])[0]
    else :
        g = f
        outputdir = os.path.split(args.filename)[0]
    
    key = params['make_probe']['h5_group']+'/P'
    if key in g :
        del g[key]
    g[key] = P_sample
    
    key = params['make_probe']['h5_group']+'/pupil'
    if key in g :
        del g[key]
    g[key] = P
    
    key = params['make_probe']['h5_group']+'/defocus'
    if key in g :
        del g[key]
    g[key] = params['make_probe']['defocus']
        
    key = params['make_probe']['h5_group']+'/Fresnel'
    if key in g :
        del g[key]
    g[key] = Fresnel
    g.close()
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print e
