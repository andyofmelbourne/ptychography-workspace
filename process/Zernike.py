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

    # phase
    # ------------------
    if params['Zernike']['orders'] is not None :
        orders = params['Zernike']['orders']
    else :
        orders = 36
    
    phase = phase_full[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    
    # Orthogonalise the Zernike polynomials 
    # -------------------------------------
    # define the x-y grid to evaluate the polynomials 
    # evaluate on our grid [2xN, 2xM] where N and M are the frame dims
    # rectangle in circle domain
    shape = (ROI[1]-ROI[0], ROI[3]-ROI[2])
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
    import scipy.constants as sc
    du = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
    z  = f['/entry_1/instrument_1/detector_1/distance'][()]
    E  = f['/entry_1/instrument_1/source_1/energy'][()]
    wavelen = sc.h * sc.c / E
    dq      = du / (wavelen * z)
    
    # get the aberration function (phi)
    #phi = - phase / (2. * np.pi * wavelen)
    phi = phase 
    
    dA = (y[1]-y[0])*(x[1]-x[0])
    z = [np.sum(dA * b * phi) for b in basis_grid]
    z = np.array(z)
    
    print '\n\n'
    print 'Zernike coefficients'
    print '--------------------'
    print 'Noll index, weight'
    for i in range(orders):
        print i+1, z[i]

    # get defocus and astigmatism 
    # ---------------------------
    # rescale y and x 
    qy = y/(y[1]-y[0]) * dq[0]
    qx = x/(x[1]-x[0]) * dq[1]
    qy, qx = np.meshgrid(qy, qx, indexing='ij')
    
    # defocus
    # -------
    phi_df = - np.pi * wavelen * 1. * (qy**2 + qx**2)
    z_df   = np.array( [np.sum(dA * b * phi_df) for b in basis_grid] )

    # astigmatism x (or fs)
    # ---------------------
    phi_ax = - np.pi * wavelen * 1. * qx**2
    z_ax   = np.array( [np.sum(dA * b * phi_ax) for b in basis_grid] )
    
    # astigmatism y (or ss)
    # ---------------------
    phi_ay = - np.pi * wavelen * 1. * qy**2
    z_ay   = np.array( [np.sum(dA * b * phi_ay) for b in basis_grid] )

    # tilt x (or fs)
    # ---------------------
    phi_tx = -2. * np.pi * 1. * qx
    z_tx   = np.array( [np.sum(dA * b * phi_tx) for b in basis_grid] )

    # tilt y (or ss)
    # ---------------------
    phi_ty = -2. * np.pi * 1. * qy
    z_ty   = np.array( [np.sum(dA * b * phi_ty) for b in basis_grid] )

    # these are weights |z| cos(theta), were theta is the angle b/w
    # (say) z and z_df
    defocus_w = np.sum( z_df * z ) / np.sqrt(np.sum(z_df**2))
    astig_x_w = np.sum( z_ax * z ) / np.sqrt(np.sum(z_ax**2))
    astig_y_w = np.sum( z_ay * z ) / np.sqrt(np.sum(z_ay**2))
    tilt_x_w  = np.sum( z_tx * z ) / np.sqrt(np.sum(z_tx**2))
    tilt_y_w  = np.sum( z_ty * z ) / np.sqrt(np.sum(z_ty**2))

    print '\n\n'
    print 'Zernike weights'
    print '---------------'
    print 'defocus       :', defocus_w
    print 'astigmatism fs:', astig_x_w
    print 'astigmatism ss:', astig_y_w
    print 'tilt fs       :', tilt_x_w
    print 'tilt ss       :', tilt_y_w
    
    # dividing again by |z_df| gives the multiples of z_df in z
    defocus = defocus_w / np.sqrt(np.sum(z_df**2))
    astig_x = astig_x_w / np.sqrt(np.sum(z_ax**2))
    astig_y = astig_y_w / np.sqrt(np.sum(z_ay**2))
    tilt_x  = tilt_x_w  / np.sqrt(np.sum(z_tx**2))
    tilt_y  = tilt_y_w  / np.sqrt(np.sum(z_ty**2))
    
    print '\n\n'
    print 'Geometric aberrations'
    print '---------------------'
    print 'defocus       :', defocus, '(m) (+ve is overfocus)'
    print 'astigmatism fs:', astig_x, '(m)'
    print 'astigmatism ss:', astig_y, '(m)'
    print 'tilt fs       :', tilt_x, '(rad) relative to centre of roi'
    print 'tilt ss       :', tilt_y, '(rad) relative to centre of roi'
    
    if params['Zernike']['remove_tilt'] is True :
        #tilt_x_phi = tilt_x * np.sum(np.array([z_tx[i] * basis_grid[i] for i in range(orders)]), axis=0)
        #tilt_y_phi = tilt_y * np.sum(np.array([z_ty[i] * basis_grid[i] for i in range(orders)]), axis=0)
        tilt_x_phi = z[1] * basis_grid[1]
        tilt_y_phi = z[2] * basis_grid[2]
        phase -= tilt_x_phi 
        phase -= tilt_y_phi 

    if params['Zernike']['remove_pedestal'] is True :
        pedestal_phi =  z[0] * basis_grid[0]
        phase       -= pedestal_phi 

    if params['Zernike']['remove_defocus'] is True :
        #defocus_phi = defocus * np.sum(np.array([z_df[i] * basis_grid[i] for i in range(orders)]), axis=0)
        defocus_phi = z[3] * basis_grid[3]
        phase      -= defocus_phi 

    if params['Zernike']['remove_astigmatism'] is True :
        #astig_x_phi = astig_x * np.sum(np.array([z_ax[i] * basis_grid[i] for i in range(orders)]), axis=0)
        #astig_y_phi = astig_y * np.sum(np.array([z_ay[i] * basis_grid[i] for i in range(orders)]), axis=0)
        astig_x_phi = z[4] * basis_grid[4]
        astig_y_phi = z[5] * basis_grid[5]
        phase -= astig_x_phi 
        phase -= astig_y_phi 

    phase_full = f[params['Zernike']['phase']][()]
    phase_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = phase
    
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
    
    # pupil
    key = params['Zernike']['h5_group']+'/phase'
    if key in g :
        del g[key]
    g[key] = phase_full

    # phase
    key = params['Zernike']['h5_group']+'/Zernike_coefficients'
    if key in g :
        del g[key]
    g[key] = z

    # aberration
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
