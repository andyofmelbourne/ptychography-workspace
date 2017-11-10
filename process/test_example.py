#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# make an example cxi file
# with a small sample and small aberations

import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'utils'))
sys.path.insert(0, os.path.join(root, 'process'))

import h5py
import scipy.misc
import utils
import numpy as np

try :
    import ConfigParser as configparser 
except ImportError :
    import configparser

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='generate an example cxi file for testing')
    parser.add_argument('filename', type=str, \
                        help="file name of the *.cxi file")
    parser.add_argument('-c', '--config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    # check that h5 file exists, if not create it
    if not os.path.exists(args.filename):
        outputdir = os.path.split(os.path.abspath(args.filename))[0]
        
        # mkdir if it does not exist
        if not os.path.exists(args.filename):
            yn = input(str(args.filename) + ' does not exist. Create it? [y]/n : ')
            print('yn:', yn)
            if yn.strip() == 'y' or yn.strip() == '' :
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                
                # make an empty file
                f = h5py.File(args.filename, 'w')
                f.close()
            else :
                raise NameError('h5 file does not exist: ' + args.filename)
    
    # if config is non then read the default from the *.cxi dir
    if args.config is None :
        args.config = os.path.join(os.path.split(args.filename)[0], 'test_example.ini')
        
        # if there is no config file in the cxi dir then read the default from the process dir
        if not os.path.exists(args.config):
            args.config = os.path.join(root, 'process/test_example.ini')
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = configparser.ConfigParser()
    config.read(args.config)
    
    params = utils.parse_parameters(config)
    
    return args, params['example']

def make_object(**kwargs):
    # transmission of the object
    O = scipy.misc.ascent().astype(np.float)
    
    # scale
    O /= O.max()

    # sample plane sampling
    dx  = kwargs['pix_size'] * kwargs['focal_length'] / kwargs['det_dist']
    
    # interpolate
    #############
    # desired grid 
    yd, xd = np.arange(0, kwargs['o_size'], dx), np.arange(0, kwargs['o_size'], dx)
    
    # current x, y values
    y, x = np.linspace(0, kwargs['o_size'], O.shape[0]), np.linspace(0, kwargs['o_size'], O.shape[1])
    
    # interpolate onto the new grid
    from scipy.interpolate import RectBivariateSpline
    rbs = RectBivariateSpline(y, x, O, bbox = [y.min(), y.max(), x.min(), x.max()], s = 1., kx=1, ky=1)
    O = rbs(yd, xd)
    return O

def make_probe(**kwargs):
    # aperture 
    shape = (128,128)
    roi   = [30, 100, 20, 108]
    #roi   = [0, 128, 0, 128]
    
    # probe
    P = np.zeros((128,128), dtype=np.float) 
    P[roi[0]:roi[1], roi[2]:roi[3]] = 1.

    # smooth the edges 
    from scipy.ndimage import gaussian_filter
    P = gaussian_filter(P, 2.0, mode='constant') + 0J
    
    # add aberrations
    """
    dx  = kwargs['pix_size']  
    y, x = np.fft.fftfreq(P.shape[0], 1./P.shape[0])*dx, np.fft.fftfreq(P.shape[1], 1./P.shape[1])*dx
    y, x = np.meshgrid(y, x, indexing='ij')
    X2 = y**2 + x**2
    X2 = np.fft.fftshift(X2)
    
    from scipy import constants as sc
    wav = sc.h * sc.c / kwargs['energy']
    ex = np.exp(-1.0J * np.pi * X2 / (wav * kwargs['det_dist']))
    P *= ex
    """
    
    back_prop = make_back_prop(P.shape, kwargs['det_dist'], kwargs['defocus'], kwargs['pix_size'], kwargs['energy'])
    
    # real-space probe
    Ps = back_prop(P)
    return Ps, P


def _make_frames(O, Ps, forward_prop, pos):
    # in pixels
    y_n, x_n = pos
    
    # make frames 
    frames = []
    i, j = np.indices(Ps.shape)
    for y in y_n :
        for x in x_n :
            ss = np.rint(i - y).astype(np.int)
            fs = np.rint(j - x).astype(np.int)
            frame = forward_prop(Ps * O[ss, fs])
            frame = np.abs(frame)**2
            frames.append(frame) 
    
    return frames

def make_forward_prop(shape, z, df, du, en):
    """
    wave_det = IFFT[ FFT[wav_sample] * e^{-i \pi \lambda z_eff (q * z / df)**2} ]
    where q_n = n z/N df du, x_n = n du
    """
    # wavelength
    from scipy import constants as sc
    wav = sc.h * sc.c / en

    #dx    = df * du / z
    dx    = du 
    z_eff = df * (z-df) / z
    qi, qj = np.fft.fftfreq(shape[0], dx), np.fft.fftfreq(shape[1], dx)
    qi, qj = np.meshgrid(qi, qj, indexing='ij')
    q2 = (z/df)**2 * (qi**2 + qj**2)
    ex = np.exp(-1.0J * np.pi * wav * z_eff * q2)

    prop = lambda x : np.fft.ifftn(np.fft.fftn(x) * ex)
    return prop

def make_back_prop(shape, z, df, du, en):
    """
    wave_det = IFFT[ FFT[wav_sample] * e^{-i \pi \lambda z_eff (q * z / df)**2} ]
    where q_n = n z/N df du, x_n = n du
    """
    # wavelength
    from scipy import constants as sc
    wav = sc.h * sc.c / en

    dx    = du 
    z_eff = df * (z-df) / z
    qi, qj = np.fft.fftfreq(shape[0], dx), np.fft.fftfreq(shape[1], dx)
    qi, qj = np.meshgrid(qi, qj, indexing='ij')
    q2 = (z/df)**2 * (qi**2 + qj**2)
    ex = np.exp(1.0J * np.pi * wav * z_eff * q2)
    
    prop = lambda x : np.fft.ifftn(np.fft.fftn(x) * ex)
    return prop

def make_frames(**kwargs):
    """
    psi_s(x)_n = Ps(x) x O(x-x_n), x_n = sample shift 
    x_n = -n (Xo - Xp)/N, 
    N = no. of positions, Xo = object size, Xp = probe size
    """
    O = make_object(**kwargs)
    
    Ps, Pd = make_probe(**kwargs)
    
    # make the sample positions
    dx  = kwargs['pix_size'] * kwargs['focal_length'] / kwargs['det_dist']
    Xp  = np.array(Ps.shape) * dx # probe dimensions
    
    if kwargs['o_size'] < np.max(Xp) :
        raise ValueError('Error: o_size is less than the probe size... Make o_size bigger than '+str(np.max(Xp)))
    
    y_n = np.linspace(0, -(kwargs['o_size'] - Xp[0]), kwargs['ny'])
    x_n = np.linspace(0, -(kwargs['o_size'] - Xp[1]), kwargs['nx'])
    
    # make the forward propagator
    
    #forward_prop = lambda x : x
    forward_prop = make_forward_prop(Ps.shape, kwargs['det_dist'], kwargs['defocus'], kwargs['pix_size'], kwargs['energy'])
     
    frames = _make_frames(O, Ps, forward_prop, (y_n/dx, x_n/dx))
    
    return Pd, Ps, frames


if __name__ == '__main__':
    args, params = parse_cmdline_args()
    
    P, Ps, frames = make_frames(**params)
