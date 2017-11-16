#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import scipy.constants as sc
import h5py
import numpy as np

import time
try :
    import ConfigParser as configparser 
except ImportError :
    import configparser

import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'utils'))
sys.path.insert(0, os.path.join(root, 'process'))

import utils 
import optics 
from numpy.polynomial import polynomial as P
from numpy.polynomial import legendre as L
from get_Fresnel_pixel_shifts_cxi import get_Fresnel_pixel_shifts_cxi

#import pyximport; pyximport.install()
#from feature_matching import feature_map_cython

class Cpu_stitcher():

    def __init__(self, data, mask, W, R, O, X_ij): 
        dtype     = np.float64
        self.IW   = mask * data * W
        self.Od   = np.zeros_like(data, dtype=dtype)
        self.mask = mask
        self.WW   = mask * W**2
        self.W    = mask * W
        self.R    = np.rint(R).astype(np.int)
        
        # add room for half a data frame
        self.R[:, 0] -= data.shape[1]//2
        self.R[:, 1] -= data.shape[2]//2
        
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
            ss = self.i + X_ij[0] - self.R[k][0]
            fs = self.j + X_ij[1] - self.R[k][1]
            mask = (ss > 0) * (ss < self.O.shape[0]) * (fs > 0) * (fs < self.O.shape[1])
            
            self.Od[k][mask] =  self.O[ss[mask], fs[mask]]#[self.i + X_ij[0] - self.R[k][0], self.j + X_ij[1] - self.R[k][1]] 
        return self.Od
    
    def inverse_map(self, X_ij, IW_weights=None):
        self.O.fill(0)
        self.WWmap.fill(0)
        
        if IW_weights is None :
            IW_weights = np.ones_like(self.IW, dtype=np.bool)
        
        for k in range(self.IW.shape[0]):
            ss = self.i + X_ij[0] - self.R[k][0]
            fs = self.j + X_ij[1] - self.R[k][1]
            mask = (ss > 0) * (ss < self.O.shape[0]) * (fs > 0) * (fs < self.O.shape[1])
            self.O[    ss[mask], fs[mask]] += self.IW[k][mask] * IW_weights[k][mask]
            self.WWmap[ss[mask], fs[mask]] += self.WW[mask] * IW_weights[k][mask]
        self.O /= (self.WWmap + 1.0e-5)
        self.O[self.O==0] = 1.
        return self.O
    
    def calc_error(self, X_ij, IW_weights=None):
        self.O       = self.inverse_map(X_ij, IW_weights=None)
        self.Od      = self.forward_map(X_ij)
        
        # sum |sqrt(I) - sqrt(I_forward)|^2
        self.error_map  = (np.sqrt(self.IW) - np.sqrt(self.Od * self.WW))**2 / \
                          (1.0e-5+self.W)
        error = np.sum( self.error_map )
        return error
    
    def cut_object(self, O, pad = 2):
        shape = (self.IW.shape[0], pad * self.IW.shape[1], pad * self.IW.shape[2])
        Os    = np.zeros( shape, dtype = O.dtype)
        i, j  = np.indices(shape[1:])
        i    -= self.IW.shape[1] * (pad - 1)//2 + 1
        j    -= self.IW.shape[2] * (pad - 1)//2 + 1
        for k in range(self.IW.shape[0]):
            ss = i - self.R[k][0]
            fs = j - self.R[k][1]
            mask = (ss > 0) * (ss < shape[1]) * (fs > 0) * (fs < shape[2])
            Os[k][mask] =  O[ss[mask], fs[mask]] 
         
        # return the slice objects such that Od[k] = Os[k][i, j]
        i, j = slice(data.shape[1]//2+1,3*data.shape[1]//2+1,1), slice(data.shape[2]//2+1,3*data.shape[2]//2+1,1)
        return Os, i, j

    def speckle_tracking_update(self, steps=8, window=16, search_window=50, max_iters=20, min_overlap=3, median_filter=None, polyfit_order=None, update_pos=False):
        from multiprocessing import Pool
        import itertools 
        try :
            from itertools import izip 
        except ImportError :
            izip = zip
          
        X_ij = self.X_ij.astype(np.float)
        mask = self.mask
        
        Os     = []
        deltas = []
        NCCs   = []
        errors = []
        
        errors.append(self.calc_error(np.rint(X_ij).astype(np.int)))
        print('Error:', errors[-1])

        Os.append(self.O.copy())
        deltas.append(X_ij.copy())
        NCCs.append(np.zeros_like(X_ij[0]))
        
        pool = Pool()
        for ii in range(max_iters):
            print('\n\nloop :', ii)
            
            print('setting up... ')
            forwards, i, j = self.cut_object(self.O)
            overlaps, i, j = self.cut_object(self.WWmap)

            # demand 3x overlap for X_ij updates
            mask_forwards  = (overlaps > (min_overlap * np.median(self.WW))).astype(np.int)
            frames         = self.IW / (1.0e-5+self.WW)
             
            # add a random offset between 0 --> steps-1
            offset_i = ( (ii + np.arange(forwards.shape[0])) // steps) % steps
            offset_j = (ii + np.arange(forwards.shape[0])) % steps
            
            print('sending to workers ')
            print(forwards.shape, frames.shape, mask.shape, window, search_window, steps)
            args = izip( frames, forwards, itertools.repeat(np.rint(X_ij).astype(np.int)), itertools.repeat(mask.astype(np.int)), \
                         mask_forwards,\
                         itertools.repeat(i.start), itertools.repeat(j.start), \
                         itertools.repeat(window), itertools.repeat(search_window), itertools.repeat(steps), \
                         offset_i, offset_j)
            
            res  = pool.map(feature_map_cython_wrap, args) 

            print('workers are done')
            nccs = np.array([i[1] for i in res])
            di   = np.array([i[0][0] for i in res], dtype=np.float)
            dj   = np.array([i[0][1] for i in res], dtype=np.float)
            
            # remove constant offsets
            #dri = np.mean(di, axis=(1,2))
            #drj = np.mean(dj, axis=(1,2))
            dri = np.array( [np.mean(di[i][offset_i[i]::steps, offset_j[i]::steps]) for i in range(frames.shape[0])] )
            drj = np.array( [np.mean(dj[i][offset_i[i]::steps, offset_j[i]::steps]) for i in range(frames.shape[0])] )
            for i in range(di.shape[0]):
                di[i, :, :]  -= dri[i]
                dj[i, :, :]  -= drj[i]
            
            # update positions if unknown (dangerous)
            if update_pos :
                self.R[:, 0] -= np.rint(1 * dri).astype(np.int)
                self.R[:, 1] -= np.rint(1 * drj).astype(np.int)
                print('updating sample positions:')
                print('Delta R (pixels):')
                print(np.rint(1 * dri).astype(np.int))
                print(np.rint(1 * drj).astype(np.int))
            
            # Merge and smooth pixel displacements
            print(nccs.shape, di.shape, dj.shape)
            # do a weigted sum
            norm = np.sum(nccs, axis=0) + 1.0e-10
            X_ij[0] += np.sum(di*nccs, axis=0) / norm
            X_ij[1] += np.sum(dj*nccs, axis=0) / norm

            if median_filter is not None :
                #from scipy.signal import medfilt2d
                #X_ij[0] = medfilt2d(X_ij[0], median_filter)
                #X_ij[1] = medfilt2d(X_ij[1], median_filter)
                import scipy.ndimage.filters
                from scipy.ndimage.filters import gaussian_filter
                X_ij[0] = gaussian_filter(X_ij[0], median_filter)
                X_ij[1] = gaussian_filter(X_ij[1], median_filter)
            
            if polyfit_order is not None :
                C_mask = (norm - np.mean(norm)) < np.std(norm)
                coeff, X_ij[0] = polyfit2d(X_ij[0], C_mask, order_ss=polyfit_order)
                coeff, X_ij[1] = polyfit2d(X_ij[1], C_mask, order_fs=polyfit_order)
            
            # Remove residual constant offsets
            X_ij[0] -= np.mean(X_ij[0])
            X_ij[1] -= np.mean(X_ij[1])
            
            # Calculate sum squared error
            errors.append(self.calc_error(np.rint(X_ij).astype(np.int), IW_weights = nccs))
            print('Error:', errors[-1])
            
            #if errors[-1] > errors[-2] and ii > 0 :
            #    break
            
            Os.append(self.O.copy())
            deltas.append(X_ij.copy())
        
        return X_ij, np.array(Os), np.array(errors), norm

def feature_map_cython_wrap(x):
    return utils.feature_map_cython(*x)

def feature_map_wrap(x):
    return feature_map(*x)

def feature_map(Od, O, X_ij, mask, i, j, window=10, search_window=20, steps=1, offset_i=0, offset_j=0):
    """
    Od = data / W # data view of the object
    O  = forward guess for the object in the region
    i, j = location of the top right courner of Od in O
    """
    X_ij_new = np.zeros_like(X_ij)
    ncc_w    = np.zeros( (search_window, search_window), dtype=np.float)
    ncc      = np.zeros( Od.shape, dtype=np.float)
    ii, jj   = np.indices(Od.shape)
    imap = ii + X_ij[0] + i
    jmap = jj + X_ij[1] + j

    for ii in range(offset_i, Od.shape[0], steps):
        for jj in range(offset_j, Od.shape[1], steps):
            # get the data segment
            i_d = slice(max(ii-window//2, 0), min(ii+window//2, Od.shape[0]), 1) 
            j_d = slice(max(jj-window//2, 0), min(jj+window//2, Od.shape[1]), 1)
            m     = mask[i_d, j_d]
            data  = m * Od[i_d, j_d]

            ncc_w.fill(0)
            # search the local area of O
            # compare 'data' with the search area in 'O'
            indices = range(-search_window//2, search_window//2, 1)
            for k in range(search_window):
                for l in range(search_window):
                    data_O = O[imap[i_d, j_d]+indices[k], jmap[i_d, j_d]+indices[l]] * m
                    
                    X, XX = np.mean(data),   np.mean(data**2)
                    Y, YY = np.mean(data_O), np.mean(data_O**2)
                    XY    = np.mean(data*data_O)
                    
                    # calculate the pearson coefficient
                    den = (np.sqrt(XX - X**2) * np.sqrt(YY - Y**2))
                    
                    # check if we are out of range
                    if den < 1.0e-5 :
                        ncc_w[k, l]   = -1.
                    else :
                        ncc_w[k, l]   = (XY - X*Y) / den
                    
            k, l  = np.unravel_index(np.argmax(ncc_w), ncc_w.shape)
            ncc_w = (ncc_w + 1.)/2.
            ncc_w = ncc_w / np.sum(ncc_w)
            ncc[ii, jj]         = ncc_w[k, l]
            X_ij_new[0][ii, jj] = X_ij[0][ii, jj] + indices[k]
            X_ij_new[1][ii, jj] = X_ij[1][ii, jj] + indices[l]

            # -------------------------------------
            # Plot the match
            # -------------------------------------
            i_im = X_ij[0][ii, jj] + indices[k] + imap[ii, jj]
            j_im = X_ij[1][ii, jj] + indices[l] + jmap[ii, jj]
            print('i, j', ii, jj, 'found match at: image (full) reference frame:', (i_im, j_im),'change in X_ij:', indices[k], indices[l], 'confidence:', ncc_w[k, l])

            import matplotlib.pyplot as plt
            image = O
            coin  = Od
            
            fig = plt.figure(figsize=(16, 6))
            ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
            #ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
            ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
            ax3 = plt.subplot(1, 3, 3, adjustable='box-forced')

            vmin = [0.8, 1.2]
            ax1.imshow(coin, interpolation='nearest', cmap='Greys_r', vmin=vmin)
            ax1.set_axis_off()
            ax1.set_title('template')
            # highlight template region
            hcoin, wcoin = i_d.stop - i_d.start, j_d.stop - j_d.start
            rect2 = plt.Rectangle((j_d.start, i_d.start), wcoin, hcoin, edgecolor='r', facecolor='none')
            ax1.add_patch(rect2)
            
            ax2.imshow(image, interpolation='nearest', cmap='Greys_r', vmin=vmin)
            ax2.set_axis_off()
            ax2.set_title('image')
            # highlight matched region
            hm, wm = imap[i_d, j_d].max() - imap[i_d, j_d].min(), jmap[i_d, j_d].max() - jmap[i_d, j_d].min() 
            rect   = plt.Rectangle((jmap[i_d, j_d].min()+indices[l], imap[i_d, j_d].min()+indices[k]), wcoin, hcoin, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)
            # highlight searched region
            hsearch, wsearch = imap[i_d, j_d].max() - imap[i_d, j_d].min() + search_window, jmap[i_d, j_d].max() - jmap[i_d, j_d].min() + search_window
            rect3 = plt.Rectangle((jmap[i_d, j_d].min()-search_window//2, imap[i_d, j_d].min()-search_window//2), wsearch, hsearch, edgecolor='g', facecolor='none')
            ax2.add_patch(rect3)
            
            ax3.imshow(ncc_w, interpolation='nearest', cmap='Greys_r')
            ax3.set_axis_off()
            ax3.set_title('`match_template`\nresult')
            # highlight matched region
            ax3.autoscale(False)
            ax3.plot(l, k, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

            plt.show()
    return X_ij_new, ncc

def polyfit2d(Z, mask, order_ss=1, order_fs=1):
    print('setting up A and B matrices...')
    A   = []
    mat = np.zeros((order_ss, order_fs), dtype=np.float)
    i = np.linspace(-1, 1, mask.shape[0])
    j = np.linspace(-1, 1, mask.shape[1])
    for ii in range(order_ss):
        for jj in range(order_fs):
            mat.fill(0)
            mat[ii, jj] = 1
            A.append(P.polygrid2d(i, j, mat)[mask])
    
    A = np.array(A).T
    B = Z[mask].flatten()
    
    print('sending to np.linalg.lstsq ...')
    coeff, r, rank, s = np.linalg.lstsq(A, B)
    print('residual:', r)
    coeff = coeff.reshape((order_ss, order_fs))
    # return the coefficients and the fit
    fit = P.polygrid2d(i, j, coeff)
    return coeff, fit


def pixel_shifts_to_phase_old(ss_shifts, fs_shifts, dx, du, df, lamb, z):
    """
    ss_shifts[i, j] = i - i_O
    fs_shifts[i, j] = j - j_O
    
    i, j are expected pixel locations of a detector speckle in the 
    object and i_O, j_O are the found locations of the speckle in 
    the object. So:
        - (I/W)[i, j] = O[i + ss_shifts[i, j] - R_i, j + ss_shifts[i, j] - R_j]  
    
    - First convert the pixel shifts to angles (radians)
      anglex_[i, j] = arctan(fs_shifts_[i, j] * dx / df) 
      angley_[i, j] = arctan(ss_shifts_[i, j] * dx / df) 
      where df = focus --> sample distance 
            dx = virtual pixel size (pixel size without phase curvature)
               = du * df / z ; pixel size * defocus / (focus --> detector)
    
    - The angles are directly related to the phase derivative at the pupil
      by:
        - (d phi / dx)_ij = - anglex_ij
        - (d phi / dy)_ij = - angley_ij
        
      where we have defined:
        pupil(x) = |A(x)| x exp( - 2 pi i / lamb * phi(x) )  (1)
        pupil(x) = |A(x)| x exp( i phase(x) )                (2)
    
      Note the minus sign in the definition for phi. 
      
    - So we need to integrate anglex and angley on the pixel grid:
        - phi_y_ij = - dx[0] * sum_k=0^i-1 angley_ij = phi_ij - phi_0j
        - phi_x_ij = - dx[1] * sum_k=0^j-1 anglex_ij = phi_ij - phi_i0

    - We can determine phi_0,j and phi_i,0 up to a constant offset with:
        - phi_y_i0 = phi_i0 - phi_00
        - phi_x_0j = phi_0j - phi_00

    - Therefore:
        - phi_ij + phi_00 = phi_y_ij + phi_x_0j 
        and the other way
        - phi_ij + phi_00 = phi_x_ij + phi_y_i0 

    - Finally I guess we should take the average:
        - phi_ij + phi_00 = (phi_y_ij + phi_x_0j + phi_x_ij + phi_y_i0) / 2.
        
      we don't care about the phi_00 factor, since it is not measurable.

    Returns
    -------
    phi : 2D numpy array
        The aberration function of the pupil as defined in (1).
    
    phase : 2D numpy array
        The phase of the pupil as defined in (2). 
    # -----------------------------------------------------
    # convert the pixel shifts to angles (radians)
    #   - anglex_[i, j] = arctan(fs_shifts_[i, j] * dx / z) 
    #   - angley_[i, j] = arctan(ss_shifts_[i, j] * dx / z) 
    # -----------------------------------------------------
    angley = np.arctan2(ss_shifts * dx[0], df)
    anglex = np.arctan2(fs_shifts * dx[1], df)
    
    # ---------------------------------------------------------------
    # So we need to integrate anglex and angley on the pixel grid:
    #    - phi_y_ij = - dx[0] * sum_k=0^i-1 angley_ij = phi_ij - phi_0j
    #    - phi_x_ij = - dx[1] * sum_k=0^j-1 anglex_ij = phi_ij - phi_i0
    # ---------------------------------------------------------------
    phi_y = - dx[0] * np.cumsum(angley, axis=0, dtype=np.float)
    phi_x = - dx[1] * np.cumsum(anglex, axis=1, dtype=np.float)
    
    # ----------------------------------------------------------------------
    # Finally I guess we should take the average:
    #   - phi_ij + phi_00 = (phi_y_ij + phi_x_0j + phi_x_ij + phi_y_i0) / 2.
    # ----------------------------------------------------------------------
    phi = phi_y + phi_x + phi_x[0, :]
    
    # this is because broadcasting is performed along the last dimension
    phi  = (phi.T + phi_y[:, 0].T).T 
    phi /= 2.

    phase = 2. * np.pi / lamb * phi
    """
    phi_y = du * np.pi / lamb * np.cumsum(ss_shifts, axis=0, dtype=np.float)
    phi_x = du * np.pi / lamb * np.cumsum(fs_shifts, axis=1, dtype=np.float)
    
    # this is because broadcasting is performed along the last dimension
    phi = phi_y + phi_x + phi_x[0, :]
    phi  = (phi.T + phi_y[:, 0].T).T 
    phi /= 2.
    
    ab = phi / (2. * np.pi / lamb)
    return ab, phi

def pixel_shifts_to_phase(delta_ij, z, du, lamb, df):
    phi_y = 2 * df * du**2 * np.pi / (lamb * z) * np.cumsum(delta_ij[0], axis=0, dtype=np.float)
    phi_x = 2 * df * du**2 * np.pi / (lamb * z) * np.cumsum(delta_ij[1], axis=1, dtype=np.float)
    
    # this is because broadcasting is performed along the last dimension
    phi = phi_y + phi_x + phi_x[0, :]
    phi  = (phi.T + phi_y[:, 0].T).T 
    phi /= 2.

    #phi -= phi[delta_ij.shape[1]//2, delta_ij.shape[2]//2]
    
    ab = phi / (2. * np.pi / lamb)
    return -ab, -phi

def get_focus_probe(P):
    # zero pad
    P2 = np.zeros( (2*P.shape[0], 2*P.shape[1]), dtype=P.dtype)
    P2[:P.shape[0], :P.shape[1]] = P
    P2 = np.roll(P2, P.shape[0]//2, 0)
    P2 = np.roll(P2, P.shape[1]//2, 1)
     
    # real-space probe
    P2 = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(P2)))
    return P2

def get_sample_plane_probe(p, lamb, z, du, df):
    # zero pad
    P2 = np.zeros( (2*p.shape[0], 2*p.shape[1]), dtype=p.dtype)
    P2[:p.shape[0], :p.shape[1]] = p
    P2 = np.roll(P2, p.shape[0]//2, 0)
    P2 = np.roll(P2, p.shape[1]//2, 1)
    
    dq = (z / df) / (du * np.array(p.shape))
    
    i = np.fft.fftfreq(P2.shape[0], 1/float(P2.shape[0])) * dq[0]
    j = np.fft.fftfreq(P2.shape[1], 1/float(P2.shape[1])) * dq[1]
    i, j = np.meshgrid(i, j, indexing='ij')
    
    exp = np.exp(-1J * lamb * df * (i**2 + j**2))

    P3 = np.fft.ifftn(np.fft.fftn( P2 ) * exp.conj())
    
    return P3


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
        args.config = os.path.join(os.path.split(args.filename)[0], 'cpu_stitch.ini')
        if not os.path.exists(args.config):
            args.config = '../process/cpu_stitch.ini'
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = configparser.ConfigParser()
    config.read(args.config)
    
    params = utils.parse_parameters(config)
    
    return args, params

def fill_pixel_shifts_from_edge(delta_ij):
    """
    assume delta_ij is zero outside a rectangular region
    """
    out = delta_ij.copy()
    # find the bottom edge
    bot = 0
    for i in range(out.shape[1]):
        if np.any(np.abs(delta_ij[0][i]) > 0):
            bot = i
            break
    # find the top edge
    top = out.shape[1]-1
    for i in range(top, 0, -1):
        if np.any(np.abs(delta_ij[0][i]) > 0):
            top = i
            break
    # find the left edge
    left = 0
    for i in range(out.shape[2]):
        if np.any(np.abs(delta_ij[0][:,i]) > 0):
            left = i
            break
    # find the right edge
    right = out.shape[2]-1
    for i in range(right, 0, -1):
        if np.any(np.abs(delta_ij[0][:,i]) > 0):
            right = i
            break
        
    #print('filling zero values in delta_ij')
    #print('edges at:', left, right, top, bot)
    
    for i in range(left):
        out[0][:, i] = np.mean(delta_ij[0][:, left:left+4], axis=-1)
        out[1][:, i] = np.mean(delta_ij[1][:, left:left+4], axis=-1)

    for i in range(out.shape[2]-1, right, -1):
        out[0][:, i] = np.mean(delta_ij[0][:, right-4:right], axis=-1)
        out[1][:, i] = np.mean(delta_ij[1][:, right-4:right], axis=-1)

    for i in range(bot):
        out[0][i, :] = np.mean(delta_ij[0][bot:bot+4, :], axis=-2)
        out[1][i, :] = np.mean(delta_ij[0][bot:bot+4, :], axis=-2)

    for i in range(out.shape[1]-1, top, -1):
        out[0][i, :] = np.mean(delta_ij[0][top-4:top, :], axis=-2)
        out[1][i, :] = np.mean(delta_ij[1][top-4:top, :], axis=-2)
    return out

if __name__ == '__main__':
    args, params = parse_cmdline_args()
    f = h5py.File(args.filename)
    
    ################################
    # Get the inputs
    # frames, df, R, O, W, ROI, mask
    ################################
    group = params['cpu_stitch']['h5_group']
    
    # ROI
    # ------------------
    if params['cpu_stitch']['roi'] is not None :
        ROI = params['cpu_stitch']['roi']
    else :
        ROI = [0, f['entry_1/data_1/data'].shape[0], 0, f['entry_1/data_1/data'].shape[1]]
    
    # frames
    # ------------------
    # get the frames to process
    if params['cpu_stitch']['good_frames'] is not None :
        good_frames = list(f[params['cpu_stitch']['good_frames']][()])
    else :
        good_frames = range(f['entry_1/data_1/data'].shape[0])
    
    data = np.array([f['/entry_1/data_1/data'][fi][ROI[0]:ROI[1], ROI[2]:ROI[3]] for fi in good_frames])
    
    # df
    # ------------------
    # get the sample to detector distance
    if params['cpu_stitch']['defocus'] is not None :
        df = params['cpu_stitch']['defocus']
    else :
        df = f['/entry_1/sample_3/geometry/translation'][0, 2]
    
    # R
    # ------------------
    # get the pixel shift coordinates along ss and fs
    R, dx = get_Fresnel_pixel_shifts_cxi(f, good_frames, params['cpu_stitch']['defocus'], offset_to_zero=True)
    
    # allow for astigmatism
    if params['cpu_stitch']['defocus_fs'] is not None :
        R[:, 1] *= df / params['cpu_stitch']['defocus_fs']
    
    # W
    # ------------------
    # get the whitefield
    if params['cpu_stitch']['whitefield'] is not None :
        W = f[params['cpu_stitch']['whitefield']][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
        if params['cpu_stitch']['whitefield'] == 'process_2/powder' :
            W /= float(f['/entry_1/data_1/data'].shape[0])
    else :
        W = np.mean(data, axis=0)

    # mask
    # ------------------
    # mask hot / dead pixels
    if params['cpu_stitch']['mask'] is None :
        if 'entry_1/instrument_1/detector_1/mask' in f:
            bit_mask = f['entry_1/instrument_1/detector_1/mask'].value
            # hot (4) and dead (8) pixels
            mask     = ~np.bitwise_and(bit_mask, 4 + 8).astype(np.bool) 
        else :
            mask     = np.ones(f['/entry_1/data_1/data'].shape[1:], dtype=np.bool)
    else :
        mask = f[params['cpu_stitch']['mask']].value
    mask     = mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]

    # delta_ij
    # -------------------
    if params['cpu_stitch']['pixel_shifts'] is not None :
        delta_ij    = np.zeros((2,) + f['/entry_1/data_1/data'].shape[1:], dtype=np.float)
        delta_ij    = f[params['cpu_stitch']['pixel_shifts']][()]
        delta_ij    = delta_ij[:, ROI[0]:ROI[1], ROI[2]:ROI[3]]
        delta_from_file = True

        # if delta_ij is zero within the ROI then extend the pixel shifts by padding
        delta_ij = fill_pixel_shifts_from_edge(delta_ij)
    else :
        delta_from_file = False
        delta_ij    = None

    f.close()
    
    W *= mask

    # apply rotation
    if 'rot_degrees' in params['cpu_stitch'] and params['cpu_stitch']['rot_degrees'] is not None :
        rot_rad = params['cpu_stitch']['rot_degrees'] * np.pi / 180.
        A     = np.array([[np.cos(rot_rad), -np.sin(0.)], [np.sin(rot_rad), np.cos(0.)]])
        #A_inv = np.array([[np.cos(tx), -np.sin(ty)], [np.sin(tx), np.cos(ty)]])
        R2    = np.dot(A, R.T).T
        
        #dx' = A . (X + dx) - X
        X_ij2     = delta_ij.copy() 
        i, j      = np.indices(delta_ij.shape[1:])
        X_ij2[0] += i
        X_ij2[1] += j
        
        X_ij2 = np.dot(A, X_ij2.reshape((2, -1))).reshape(delta_ij.shape)
        
        X_ij2[0] -= i
        X_ij2[1] -= j
        
        R2 = np.rint(R2).astype(np.int)
        cpu_stitcher = Cpu_stitcher(data, mask, W, R2, None, X_ij2)
    else :
        R = np.rint(R).astype(np.int)
        cpu_stitcher = Cpu_stitcher(data, mask, W, R, None, delta_ij)
    
    if params['cpu_stitch']['fit_grads'] :
        delta_ij, Os, errors, C_pear = cpu_stitcher.speckle_tracking_update(steps=params['cpu_stitch']['steps'], \
                                                                    window=params['cpu_stitch']['window'], \
                                                                    search_window=params['cpu_stitch']['search_window'], \
                                                                    max_iters=params['cpu_stitch']['max_iters'], \
                                                                    min_overlap=params['cpu_stitch']['min_overlap'], \
                                                                    median_filter=params['cpu_stitch']['median_filter'], \
                                                                    polyfit_order=params['cpu_stitch']['polyfit_order'], \
                                                                    update_pos=params['cpu_stitch']['update_positions'])
    else :
        print('stitching...')
        Os       = [cpu_stitcher.inverse_map(cpu_stitcher.X_ij)]
        errors   = [0]
        C_pear   = None


    print('Object Field of view:', np.array(Os[-1].shape) * dx)
    print('Object shape:        ', Os[-1].shape)
    print('Virtual Pixel size:  ', dx)

    f = h5py.File(args.filename)
    
    # Convert sample coordinates from pixels to meters
    ##################################################
    if params['cpu_stitch']['update_positions'] :
        R_out = utils.get_Fresnel_pixel_shifts_cxi_inverse(cpu_stitcher.R, f, good_frames, params['cpu_stitch']['defocus'], offset_to_zero=True, remove_affine=True)
    else : 
        R_out = None

    # get the phase
    ###############
    import scipy.constants as sc
    du = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
    z  = f['/entry_1/instrument_1/detector_1/distance'][()]
    E  = f['/entry_1/instrument_1/source_1/energy'][()]
    wavelen = sc.h * sc.c / E
    
    if 'rot_degrees' in params['cpu_stitch'] and params['cpu_stitch']['rot_degrees'] is not None :
        aberrations, phase = pixel_shifts_to_phase(X_ij2, z, 75.0e-6, wavelen, df)
    else :
        aberrations, phase = pixel_shifts_to_phase(delta_ij, z, 75.0e-6, wavelen, df)
    pupil              = np.sqrt(W) * np.exp(1J * phase)
    
    # put back into det frame
    #########################
    delta_ij_full = np.zeros((2,) + f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    delta_ij_full[0][ROI[0]:ROI[1], ROI[2]:ROI[3]] = delta_ij[0]
    delta_ij_full[1][ROI[0]:ROI[1], ROI[2]:ROI[3]] = delta_ij[1]
    
    phase_full = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    phase_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = phase
    
    aberrations_full = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    aberrations_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = aberrations
    
    pupil_full = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.complex)
    pupil_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = pupil
    
    W_full = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    W_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = W
    
    # get the sample plane probe
    # ray tracing
    ############################
    W_ray_tracing     = np.zeros((2*W.shape[0], 2*W.shape[1]), dtype=np.complex128)
    phase_ray_tracing = np.zeros((2*W.shape[0], 2*W.shape[1]), dtype=np.complex128)
    
    W_ray_tracing[cpu_stitcher.i + cpu_stitcher.X_ij[0] + W.shape[0]//2, \
                  cpu_stitcher.j + cpu_stitcher.X_ij[1] + W.shape[1]//2] = W
    
    phase_ray_tracing[cpu_stitcher.i + cpu_stitcher.X_ij[0] + W.shape[0]//2, \
                      cpu_stitcher.j + cpu_stitcher.X_ij[1] + W.shape[1]//2] = np.exp(1J * phase)
    
    P_ray_tracing = np.sqrt(W_ray_tracing) * phase_ray_tracing
    
    # get the focus spot
    ####################
    P_focus = get_focus_probe(pupil)
    
    # get the sample plane probe
    ############################
    P_sample = get_sample_plane_probe(pupil, wavelen, z, du, df)
    
    # write the result 
    ##################
    if params['cpu_stitch']['output_file'] is not None :
        g = h5py.File(params['cpu_stitch']['output_file'])
        outputdir = os.path.split(params['cpu_stitch']['output_file'])[0]
    else :
        g = f
        outputdir = os.path.split(args.filename)[0]
    
    if group not in g:
        print(g.keys())
        g.create_group(group)

    print('\nwriting to file:')
    
    # Positions
    if params['cpu_stitch']['update_positions'] :
        key = params['cpu_stitch']['h5_group']+'/R'
        if key in g :
            del g[key]
        g[key] = R_out

    # pupil
    key = params['cpu_stitch']['h5_group']+'/pupil'
    if key in g :
        del g[key]
    g[key] = pupil_full

    # phase
    key = params['cpu_stitch']['h5_group']+'/phase'
    if key in g :
        del g[key]
    g[key] = phase_full

    # aberration
    key = params['cpu_stitch']['h5_group']+'/abberation'
    if key in g :
        del g[key]
    g[key] = aberrations_full

    # ray_tracing
    key = params['cpu_stitch']['h5_group']+'/probe_sample_ray'
    if key in g :
        del g[key]
    g[key] = P_ray_tracing

    # sample plane probe propagation
    key = params['cpu_stitch']['h5_group']+'/probe_sample_fres'
    if key in g :
        del g[key]
    g[key] = P_sample

    # focal spot
    key = params['cpu_stitch']['h5_group']+'/probe_focus'
    if key in g :
        del g[key]
    g[key] = P_focus

    # errors
    if len(errors) > 1 :
        key = params['cpu_stitch']['h5_group']+'/errors'
        if delta_from_file is True :
            errors = [g[key][i] for i in range(g[key].shape[0])] + list(errors[1:])
        if key in g :
            del g[key]
        g[key] = np.array(errors)

    # object history
    if len(Os) > 1:
        key = params['cpu_stitch']['h5_group']+'/Os'
        if delta_from_file is True and key in g and Os.shape[1:] == g[key].shape[1:] :
            Os = [g[key][i] for i in range(g[key].shape[0])] + list(Os[1:])  
        if key in g :
            del g[key]
        g[key] = np.array(Os)
    
    # pixel shifts
    key = params['cpu_stitch']['h5_group']+'/pixel_shifts'
    if key in g :
        del g[key]
    g[key] = delta_ij_full
    
    # Pearson coefficients 
    if C_pear is not None :
        key = params['cpu_stitch']['h5_group']+'/C_pearson'
        if key in g :
            del g[key]
        g[key] = C_pear

    # object
    key = params['cpu_stitch']['h5_group']+'/O'
    if key in g :
        del g[key]
    g[key] = Os[-1] #np.sqrt(O).astype(np.complex128)
    
    # whitefield
    key = params['cpu_stitch']['h5_group']+'/whitefield'
    if key in g :
        del g[key]
    g[key] = W_full
    
    g.close()
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print(e)
