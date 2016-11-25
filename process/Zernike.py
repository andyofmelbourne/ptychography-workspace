"""
Fit Zernike polynomials to the aberration function phi(q):
    pupil(x) = |A(x)| x exp( - 2 pi i / lamb * phi(x) )
    pupil(q) = |A(q)| x exp( - 2 pi i lamb   * phi(q) )

First orthonormalise phi over the region of interest (roi) 
for a given number of orders. So that: 
    \sum_x Z_i(x) Z_j(x) = \delta_{i-j}

In discritised form this will be:
    \sum_n \sum_m dx[0]*dx[1] Z_i_nm Z_j_nm = \delta_{i-j}

Then use the above property to find the Zernike decomposition 
of phi(x):
    z_i = \sum_x Z_i(x) phi(x)

where Z_i(x) is the modified Zernike polynomial with Noll index i.
Now we can represent the aberration function as the sum of Zernike
polynomials:
    phi(x) = \sum_i z_i Z_i(x)

How about we take defocus, given dq, we know what defocus looks like:
    phi_df(q) = 1/2 df q^2
    phi_df_ij = 1/2 df dq^2 (i^2 + j^2)

This will have some Zernike decomposition, something like:
    phi_(df=1m)(q) = a Z_1(q) + b Z_4(q) + ...

so df = a z_1 + b z_4 + ... and so on for astigmatism (in x and y). 
"""

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
import optics 
from numpy.polynomial import polynomial as P
from numpy.polynomial import legendre as L

def variance_minimising_subtraction(f, g):
    """
    find min(f - a * g)|_a
    """
    fm = np.mean(f)
    gm = np.mean(g)
    a = np.sum( (g - gm)*(f - fm) ) / np.sum( (g - gm)**2 )
    return a


def get_focus_probe(P):
    # zero pad
    P2 = np.zeros( (2*P.shape[0], 2*P.shape[1]), dtype=P.dtype)
    P2[:P.shape[0], :P.shape[1]] = P
    P2 = np.roll(P2, P.shape[0]//2, 0)
    P2 = np.roll(P2, P.shape[1]//2, 1)
     
    # real-space probe
    P2 = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(P2)))
    return P2


def calculate_Zernike_coeff(phase, orders, dq, basis=None, basis_grid=None, y=None, x=None):
    # Orthogonalise the Zernike polynomials 
    # -------------------------------------
    # define the x-y grid to evaluate the polynomials 
    # evaluate on our grid [2xN, 2xM] where N and M are the frame dims
    # rectangle in circle domain
    if basis is None :
        shape = phase.shape
        rat = float(shape[0])/float(shape[1])
        x   = np.sqrt(1. / (1. + rat**2))
        y   = rat * x
        dom = [-y, y, -x, x]
        roi = shape
        y_vals  = np.linspace(dom[0], dom[1], shape[0])
        x_vals  = np.linspace(dom[2], dom[3], shape[1])
        
        basis, basis_grid, y, x = optics.fit_Zernike.make_Zernike_basis(\
                                  np.ones_like(phase).astype(np.bool), \
                                  roi = None, max_order = orders, return_grids = True, \
                                  yx_bounds = dom, test = False)
    
    # get the Zernike coefficients
    # ----------------------------
    phi = phase 
    
    dA = (y[1]-y[0])*(x[1]-x[0])
    z = [np.sum(dA * b * phi) for b in basis_grid]
    z = np.array(z)

    # get the Zernike fit in a polynomial basis
    z_poly = np.sum( [z[i] * basis[i] for i in range(len(basis))], axis=0)

    print '\n\n'
    print 'Zernike coefficients'
    print '--------------------'
    print 'Noll index, weight'
    for i in range(orders):
        print i+1, z[i]
    return z, z_poly, basis, basis_grid, y, x

def get_geometric_aberrations(phase, y, x, dq, wavelen, \
        remove_piston      = False, \
        remove_tilt        = False, \
        remove_astigmatism = False, \
        remove_defocus     = False):
    
    # rescale y and x 
    dA = (y[1]-y[0])*(x[1]-x[0])
    qy = y/(y[1]-y[0]) * dq[0]
    qx = x/(x[1]-x[0]) * dq[1]
    qy, qx = np.meshgrid(qy, qx, indexing='ij')
    
    # find the geometric aberrations by performing 
    # a variance minimising subtraction of each of 
    # the aberration terms
    # - remove the aberrations as we go
    
    print '\nCalculating and removing geometric aberrations:'
    print 'variance of phase:', np.var(phase)
    
    # defocus
    # -------
    phi_df = - np.pi * wavelen * 1. * (qy**2 + qx**2)
    phi_fx = - np.pi * wavelen * 1. * qx**2
    phi_fy = - np.pi * wavelen * 1. * qy**2
    defocus   = variance_minimising_subtraction(phase, phi_df)
    defocus_x = variance_minimising_subtraction(phase, phi_fx)
    defocus_y = variance_minimising_subtraction(phase, phi_fy)
    
    if remove_defocus :
        phase -= defocus * phi_df
        print '\nRemoving defocus', defocus
        print 'variance of phase:', np.var(phase)

    # astigmatism 
    # ---------------------
    phi_as = - np.pi * wavelen * 1. * (qx**2 - qy**2)
    astigmatism = variance_minimising_subtraction(phase, phi_as)

    if remove_astigmatism :
        phase -= astigmatism * phi_as
        print '\nRemoving astigmatism', astigmatism
        print 'variance of phase:', np.var(phase)

    # tilt x (or fs)
    # ---------------------
    phi_tx = -2. * np.pi * 1. * qx
    tilt_x = variance_minimising_subtraction(phase, phi_tx)
    
    if remove_tilt :
        phase -= tilt_x * phi_tx
        print '\nRemoving tilt_x', tilt_x
        print 'variance of phase:', np.var(phase)

    # tilt y (or ss)
    # ---------------------
    phi_ty = -2. * np.pi * 1. * qy
    tilt_y = variance_minimising_subtraction(phase, phi_ty)
    
    if remove_tilt :
        phase -= tilt_y * phi_ty
        print '\nRemoving tilt_y', tilt_y
        print 'variance of phase:', np.var(phase)

    # piston
    # ---------------------
    piston = np.mean(phase)
    
    if remove_piston :
        phase -= piston
        print '\nRemoving piston', piston
        print 'variance of phase:', np.var(phase)
    
    
    print '\n\n'
    print 'Geometric aberrations'
    print '---------------------'
    print 'defocus       :', defocus, '(m) (+ve is overfocus)'
    print 'defocus fs    :', defocus_x, '(m)'
    print 'defocus ss    :', defocus_y, '(m)'
    print 'astigmatism   :', astigmatism, '(m)'
    print 'tilt fs       :', tilt_x, '(rad) relative to centre of roi'
    print 'tilt ss       :', tilt_y, '(rad) relative to centre of roi'

    return phase

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='calculate the Zernike coefficients of the pupil aberration function')
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
        args.config = os.path.join(os.path.split(args.filename)[0], 'Zernike.ini')
        if not os.path.exists(args.config):
            args.config = '../process/Zernike.ini'
    
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
    group = params['Zernike']['h5_group']
    
    # ROI
    # ------------------
    if params['Zernike']['roi'] is not None :
        ROI = params['Zernike']['roi']
    else :
        ROI = [0, f['entry_1/data_1/data'].shape[0], 0, f['entry_1/data_1/data'].shape[1]]
    
    # phase
    # ------------------
    if params['Zernike']['phase'] is not None :
        phase_full = f[params['Zernike']['phase']][()]
    else :
        phase_full = None

    # Zernike orders
    # ------------------
    if params['Zernike']['orders'] is not None :
        orders = params['Zernike']['orders']
    else :
        orders = 36
    
    # W
    # ------------------
    # get the whitefield
    if params['Zernike']['whitefield'] is not None :
        W = f[params['Zernike']['whitefield']][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
        
        if params['Zernike']['whitefield'] == 'process_2/powder' :
            W /= float(f['/entry_1/data_1/data'].shape[0])
    else :
        W = None
    
    phase = phase_full[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    
    # calculate the Zernike coefficients
    # ----------------------------------
    # calcualte dq
    import scipy.constants as sc
    du      = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], \
                        f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
    z       = f['/entry_1/instrument_1/detector_1/distance'][()]
    E       = f['/entry_1/instrument_1/source_1/energy'][()]
    wavelen = sc.h * sc.c / E
    dq      = du / (wavelen * z)
    z, z_poly, basis, basis_grid, y, x = calculate_Zernike_coeff(phase, orders, dq)

    # get defocus, astigmatism and tilt
    # ---------------------------------
    phase = get_geometric_aberrations(phase, y, x, dq, wavelen, \
            remove_piston      = params['Zernike']['remove_piston'], \
            remove_tilt        = params['Zernike']['remove_tilt'], \
            remove_astigmatism = params['Zernike']['remove_astigmatism'], \
            remove_defocus     = params['Zernike']['remove_defocus'])
    
    # calculate the Zernike again
    # ----------------------------------
    z, z_poly, basis, basis_grid, y, x = calculate_Zernike_coeff(phase, orders, dq, basis, basis_grid, y, x)
    
    # make the Zernike fit
    # ---------------------------------
    phase_zern = np.sum( [z[i] * basis_grid[i] for i in range(len(basis))], axis=0)
    
    phase_full = f[params['Zernike']['phase']][()]
    phase_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = phase
    
    phase_zern_full = f[params['Zernike']['phase']][()]
    phase_zern_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = phase_zern


    # get the focus spot
    if W is not None :
        pupil = W * np.exp(1J * phase)
        P_focus = get_focus_probe(pupil)

    # write the result 
    ##################
    if params['Zernike']['output_file'] is not None :
        g = h5py.File(params['Zernike']['output_file'])
        outputdir = os.path.split(params['Zernike']['output_file'])[0]
    else :
        g = f
        outputdir = os.path.split(args.filename)[0]
    
    if group not in g:
        print g.keys()
        g.create_group(group)
    
    # focus probe
    key = params['Zernike']['h5_group']+'/probe_focus'
    if key in g :
        del g[key]
    g[key] = P_focus

    # phase
    key = params['Zernike']['h5_group']+'/phase'
    if key in g :
        del g[key]
    g[key] = phase_full

    # Zernike fit to the phase
    key = params['Zernike']['h5_group']+'/Zernike_phase_fit'
    if key in g :
        del g[key]
    g[key] = phase_zern_full

    # Zernike coefficients
    key = params['Zernike']['h5_group']+'/Zernike_coefficients'
    if key in g :
        del g[key]
    g[key] = z

    # Zernike basis functions
    key = params['Zernike']['h5_group']+'/Zernike_basis_vectors'
    if key in g :
        del g[key]
    g[key] = basis_grid
    
    g.close()
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print e
