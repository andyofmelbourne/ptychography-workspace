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
from numpy.polynomial import polynomial as P
from numpy.polynomial import legendre as L

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=' ')
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
        args.config = os.path.join(os.path.split(args.filename)[0], 'make_pixel_shifts.ini')
        if not os.path.exists(args.config):
            args.config = '../process/make_pixel_shifts.ini'
    
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
    # frames, df, R, O, W, ROI, mask
    ################################
    
    # ROI
    # ------------------
    if params['make_pixel_shifts']['roi'] is not None :
        ROI = params['make_pixel_shifts']['roi']
    else :
        ROI = [0, f['entry_1/data_1/data'].shape[0], 0, f['entry_1/data_1/data'].shape[1]]
    
    if params['make_pixel_shifts']['plot'] is not None :
        plot = f[params['make_pixel_shifts']['plot']+'/pixel_shifts'][:, ROI[0]:ROI[1],ROI[2]:ROI[3]]
    else :
        plot = None
    f.close()
    
    delta_ij = np.zeros((2, ROI[1]-ROI[0], ROI[3]-ROI[2]), dtype=np.float)

    i, j = np.arange(delta_ij.shape[1]), np.arange(delta_ij.shape[2])
    #dfs = params['make_pixel_shifts']['scale'] * np.polyval(params['make_pixel_shifts']['poly_fs'][::-1], j)
    #dss = params['make_pixel_shifts']['scale'] * np.polyval(params['make_pixel_shifts']['poly_ss'][::-1], i)
    deg = params['make_pixel_shifts']['poly_fs'][0]
    x = params['make_pixel_shifts']['poly_fs'][1::2]
    y = params['make_pixel_shifts']['poly_fs'][2::2]
    print('\n')
    print(x)
    print(y)
    print(deg)
    p = np.polyfit(x, y, deg)

    print(np.polyval(p, x))
    # now integrate 
    #p = np.polyint(p)
    dfs = params['make_pixel_shifts']['scale'] * np.polyval(p, j)

    deg = params['make_pixel_shifts']['poly_ss'][0]
    x = params['make_pixel_shifts']['poly_ss'][1::2]
    y = params['make_pixel_shifts']['poly_ss'][2::2]
    p = np.polyfit(x, y, deg)
    
    print(np.polyval(p, x))
    # now integrate 
    #p = np.polyint(p)
    dss = params['make_pixel_shifts']['scale'] * np.polyval(p, i)

    if params['make_pixel_shifts']['subtract_mean'] :
        dfs -= np.mean(dfs)
        dss -= np.mean(dss)
    
    delta_ij[0] = dss[:, None]
    delta_ij[1] = dfs[None, :]

    if plot is not None :
        dfs2 = np.mean(plot[1], axis=0)
        dss2 = np.mean(plot[0], axis=1)

    f = h5py.File(args.filename)
    
    # Convert sample coordinates from pixels to meters
    
    # put back into det frame
    #########################
    delta_ij_full = np.zeros((2,) + f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    delta_ij_full[0][ROI[0]:ROI[1], ROI[2]:ROI[3]] = delta_ij[0]
    delta_ij_full[1][ROI[0]:ROI[1], ROI[2]:ROI[3]] = delta_ij[1]
    
    # write the result 
    ##################
    if params['make_pixel_shifts']['output_file'] is not None :
        g = h5py.File(params['make_pixel_shifts']['output_file'])
        outputdir = os.path.split(params['make_pixel_shifts']['output_file'])[0]
    else :
        g = f
        outputdir = os.path.split(args.filename)[0]
    
    group = params['make_pixel_shifts']['h5_group']
    if group not in g:
        print g.keys()
        g.create_group(group)

    print '\nwriting to file:'
    # pixel shifts
    key = group + '/pixel_shifts' 
    if key in g :
        del g[key]
    g[key] = delta_ij_full
    
    key = group + '/1d_profile_fs' 
    if key in g :
        del g[key]
    g[key] = dfs    
    
    key = group + '/1d_profile_ss' 
    if key in g :
        del g[key]
    g[key] = dss
    
    if plot is not None :
        key = group + '/1d_profile_plot_fs' 
        if key in g :
            del g[key]
        g[key] = dfs2   
        
        key = group + '/1d_profile_plot_ss' 
        if key in g :
            del g[key]
        g[key] = dss2

    g.close()
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print e
