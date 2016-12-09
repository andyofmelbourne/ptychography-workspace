import scipy.constants as sc
import h5py
import numpy as np

import sys, os

import time
import ConfigParser

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(root)
sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'process'))

from Ptychography import utils as Putils
import utils 

def get_propagation_series(p, dq, lamb, dfs):
    # zero pad
    P2 = np.zeros( (2*p.shape[0], 2*p.shape[1]), dtype=p.dtype)
    P2[:p.shape[0], :p.shape[1]] = p
    P2 = np.roll(P2, p.shape[0]//2, 0)
    P2 = np.roll(P2, p.shape[1]//2, 1)
    
    i = np.fft.fftfreq(P2.shape[0], 1/float(P2.shape[0])) * dq[0]
    j = np.fft.fftfreq(P2.shape[1], 1/float(P2.shape[1])) * dq[1]
    i, j = np.meshgrid(i, j, indexing='ij')
    
    probes = np.zeros((len(dfs), P2.shape[0], P2.shape[1]), dtype=np.complex128)
    print 'propagating:'
    for ii, df in enumerate(dfs) :
        print ii, len(dfs), df
        exp = np.exp(-1J * lamb * df * (i**2 + j**2))
        probes[ii] = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(P2) * exp))
    return probes

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='calculate the propagation profile of the probe given its pupil function')
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
        args.config = os.path.join(os.path.split(args.filename)[0], 'probe_profiler.ini')
        if not os.path.exists(args.config):
            args.config = '../process/probe_profiler.ini'
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = Putils.parse_parameters(config)
    
    return args, params

if __name__ == '__main__':
    args, params = parse_cmdline_args()
    f = h5py.File(args.filename)
    
    ################################
    # Get the inputs
    # ROI, phase, orders, dq
    ################################
    group = params['probe_profiler']['h5_group']
    
    # ROI
    # ------------------
    if params['probe_profiler']['roi'] is not None :
        ROI = params['probe_profiler']['roi']
    else :
        ROI = [0, f['entry_1/data_1/data'].shape[0], 0, f['entry_1/data_1/data'].shape[1]]
    
    # phase
    # ------------------
    if params['probe_profiler']['phase'] is not None :
        phase_full = f[params['probe_profiler']['phase']][()]
    else :
        phase_full = None

    # W
    # ------------------
    # get the whitefield
    if params['probe_profiler']['whitefield'] is not None :
        whitefield_full = f[params['probe_profiler']['whitefield']][()].astype(np.float)
        
        if params['probe_profiler']['whitefield'] == 'process_2/powder' :
            whitefield_full /= float(f['/entry_1/data_1/data'].shape[0])
    else :
        whitefield_full = None

    # pupil
    # ------------------
    if params['probe_profiler']['pupil'] is not None :
        pupil_full = f[params['probe_profiler']['pupil']][()]
    else :
        pupil_full = None
    
    if pupil_full is None :
        pupil_full = np.sqrt(whitefield_full) * np.exp(1J * phase_full)
    
    pupil = pupil_full[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    
    # calcualte dq
    # ----------------------------------
    import scipy.constants as sc
    du      = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], \
                        f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
    z       = f['/entry_1/instrument_1/detector_1/distance'][()]
    E       = f['/entry_1/instrument_1/source_1/energy'][()]
    wavelen = sc.h * sc.c / E
    dq      = du / (wavelen * z)
    

    # set defocus distances
    # ----------------------------------
    start, stop, N = params['probe_profiler']['df_start_stop_n'] 
    
    dfs, df_step = np.linspace(start, stop, int(N), retstep=True)
    
    # these are zero padded
    probes = get_propagation_series(pupil, dq, wavelen, dfs)
    
    dims = [dfs[-1] - dfs[0], 1./dq[0], 1./dq[1]]
    
    # write the result 
    ##################
    if params['probe_profiler']['output_file'] is not None :
        g = h5py.File(params['probe_profiler']['output_file'])
        outputdir = os.path.split(params['probe_profiler']['output_file'])[0]
    else :
        g = f
        outputdir = os.path.split(args.filename)[0]
    
    if group not in g:
        print g.keys()
        g.create_group(group)
    
    # probe stack
    key = params['probe_profiler']['h5_group']+'/probe_stack'
    if key in g :
        del g[key]
    g[key] = probes
    
    # dimensions
    key = params['probe_profiler']['h5_group']+'/dims'
    if key in g :
        del g[key]
    g[key] = np.array(dims)

    g.close()
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print e

