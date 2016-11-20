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
import optics 
from numpy.polynomial import polynomial as P
from numpy.polynomial import legendre as L

class Test2():
    def __init__(self, data, mask, W, R, O, delta_ij): 
        """
        """
        dtype = np.float64
        self.dtype = dtype
        self.R    = R
        self.R_g  = ap.array(R)
        
        # define the x-y grid to evaluate the polynomials 
        # evaluate on our grid [2xN, 2xM] where N and M are the frame dims
        # rectangle in circle domain
        shape = (2*data.shape[1], 2*data.shape[2])
        rat = float(shape[0])/float(shape[1])
        x   = np.sqrt(1. / (1. + rat**2))
        y   = rat * x
        dom = [-y, y, -x, x]
        roi = shape
        self.y   = np.linspace(dom[0], dom[1], shape[0])
        self.x   = np.linspace(dom[2], dom[3], shape[1])
        
        # add pixel values
        self.i, self.j = np.indices(self.y.shape + self.x.shape)

        self.delta_ij  = np.array([np.zeros_like(self.i), np.zeros_like(self.j)]).astype(dtype)

        if delta_ij is not None :
            self.delta_ij[:, :data.shape[1], :data.shape[2]] = delta_ij
            self.delta_ij    = np.roll(self.delta_ij, data.shape[1]/2, axis=1)
            self.delta_ij    = np.roll(self.delta_ij, data.shape[2]/2, axis=2)
        
        self.mask = np.zeros(shape, dtype=dtype)
        self.mask[:mask.shape[0], :mask.shape[1]] = mask
        self.mask    = np.roll(self.mask, data.shape[1]/2, axis=0)
        self.mask    = np.roll(self.mask, data.shape[2]/2, axis=1)
        self.mask_g  = ap.array(self.mask)
        
        self.W    = np.zeros(shape, dtype=dtype)
        self.W[:W.shape[0], :W.shape[1]] = W * mask
        self.W    = np.roll(self.W, data.shape[1]/2, axis=0)
        self.W    = np.roll(self.W, data.shape[2]/2, axis=1)
        self.W_g  = ap.array(self.W)
        self.WW_g = self.W_g * self.W_g
        
        self.WW_lvl = np.median( (W*W*mask)[W > 0] )

        self.data2   = np.zeros( (data.shape[0],)+shape, dtype=dtype)
        self.data2[:, :data.shape[1], :data.shape[2]] = data * mask
        self.data2   = np.roll(self.data2, data.shape[1]/2, axis=1)
        self.data2   = np.roll(self.data2, data.shape[2]/2, axis=2)
        self.data2_g     = ap.array(self.data2 * self.W, dtype=dtype)
        self.whitefields = ap.zeros(self.data2_g.shape, dtype=dtype)

        # the regular pixel values
        i, j = np.indices(self.data2.shape[1 :])
         
        # make the object grid
        Oi =  i.max() + np.max(np.abs(R[:, 0]))
        Oj =  j.max() + np.max(np.abs(R[:, 1]))
        
        if O is None :
            O = np.zeros((int(round(Oi)), int(round(Oj))), dtype=dtype)
        self.O_g     = ap.array(O)
        self.norm_g  = ap.zeros(self.O_g.shape, dtype=dtype)
        Oi, Oj       = np.indices(O.shape)
        self.Oi_g, self.Oj_g = ap.array(Oi.ravel().astype(dtype)), ap.array(Oj.ravel().astype(dtype))
    
        # make a bunch of translation matrices for each frame
        self.tfs = []
        for k in range(self.data2_g.shape[0]):
            self.tfs.append(ap.array([[1, 0, self.R[k, 1]], [0, 1, self.R[k, 0]]], dtype=np.float32).d_array)

        self.tfs_2    = np.zeros((self.data2_g.shape[0], 2, 3), dtype=np.float32)
        self.tfs_2[:] = np.array([[1,0,0],[0,1,0]])
        self.tfs_2[:, 0, 2] = -self.R[:, 1]
        self.tfs_2[:, 1, 2] = -self.R[:, 0]
        self.tfs_2    = ap.array(self.tfs_2).d_array


    def data_to_O(self, delta_ij):
        self.ii, self.jj = self.i + delta_ij[0], self.j + delta_ij[1]
        
        self.ii_g, self.jj_g = ap.array(self.ii.ravel()), ap.array(self.jj.ravel())
        
        # data * W --> undistorted frames
        self.data3_g = ap.array(af.approx2(self.data2_g.d_array, self.jj_g.d_array, self.ii_g.d_array)).reshape(self.data2_g.shape)
        
        # Whitefield --> undistorted frame
        self.W2_g = ap.array(af.approx2(self.WW_g.d_array, self.jj_g.d_array, self.ii_g.d_array)).reshape(self.W_g.shape)
        
        # tiling to object frame
        self.O_g    *= 0
        self.norm_g *= 0

        self.O_g.d_array    = af.transform(self.data3_g[0].d_array, self.tfs[0], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
        self.norm_g.d_array = af.transform(self.W2_g.d_array,       self.tfs[0], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
        
        for k in range(1, self.data3_g.shape[0]):
            self.O_g.d_array    += af.transform(self.data3_g[k].d_array, self.tfs[k], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
            self.norm_g.d_array += af.transform(self.W2_g.d_array, self.tfs[k], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
            
        # normalisation by white field
        self.O_g = self.O_g/(1.0e-5+self.norm_g)
        self.O_g[self.O_g == 0] = 1.
        return self.O_g

    def data_to_O_min_mem(self, delta_ij):
        self.ii, self.jj = self.i + delta_ij[0], self.j + delta_ij[1]
        
        self.ii_g, self.jj_g = ap.array(self.ii.ravel().astype(self.dtype)), ap.array(self.jj.ravel().astype(self.dtype))
        
        # data * W --> undistorted frames
        for i in range(self.data2_g.shape[0]):
            self.data2_g[i] = ap.array(af.approx2(self.data2_g[i].d_array, self.jj_g.d_array, self.ii_g.d_array)).reshape(self.data2_g[i].shape)
        
        # Whitefield --> undistorted frame
        self.W2_g = ap.array(af.approx2(self.WW_g.d_array, self.jj_g.d_array, self.ii_g.d_array)).reshape(self.W_g.shape)
        
        # tiling to object frame
        self.O_g    *= 0
        self.norm_g *= 0

        self.O_g.d_array    = af.transform(self.data2_g[0].d_array, self.tfs[0], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
        self.norm_g.d_array = af.transform(self.W2_g.d_array,       self.tfs[0], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
        
        for k in range(1, self.data2_g.shape[0]):
            self.O_g.d_array    += af.transform(self.data2_g[k].d_array, self.tfs[k], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
            self.norm_g.d_array += af.transform(self.W2_g.d_array, self.tfs[k], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
            
        # normalisation by white field
        self.O_g = self.O_g/(1.0e-5+self.norm_g)
        self.O_g[self.O_g == 0] = 1.
        return self.O_g

    def O_to_data(self, delta_ij):
        # O to undistorted frames
        self.data3_g  = ap.array(af.transform(self.O_g.d_array, self.tfs_2, odim0=self.data3_g.shape[2], odim1=self.data3_g.shape[1])).reshape(self.data2_g.shape)

        # testing
        self.overlaps = ap.array(af.transform(self.norm_g.d_array, self.tfs_2, odim0=self.data3_g.shape[2], odim1=self.data3_g.shape[1])).reshape(self.data2_g.shape)
        
        # undistorted to distorted frames 
        self.ii, self.jj     = self.i - delta_ij[0], self.j - delta_ij[1]
        self.ii_g, self.jj_g = ap.array(self.ii.ravel()), ap.array(self.jj.ravel())
        self.data3_g         = ap.array(af.approx2(self.data3_g.d_array, self.jj_g.d_array, self.ii_g.d_array, off_grid=1.0)).reshape(self.data2_g.shape)
        #self.data3_g        /= 1.0e-3 + self.WW_g.reshape((1, self.W_g.shape[0], self.W_g.shape[1]))
        self.data3_g        *= self.WW_g.reshape((1, self.W_g.shape[0], self.W_g.shape[1]))
        
        self.overlaps     = ap.array(af.approx2(self.overlaps.d_array, self.jj_g.d_array, self.ii_g.d_array, off_grid=1.0)).reshape(self.data2_g.shape)
        return self.data3_g

    def calc_error(self, delta_ij):
        self.O_g     = self.data_to_O(delta_ij)
        self.data3_g = self.O_to_data(delta_ij)
        
        # sum |sqrt(I) - sqrt(I_forward)|^2
        self.error_map  = self.mask_g.reshape((1, self.mask_g.shape[0], self.mask_g.shape[1])) * \
                          (self.overlaps > 2.5 * self.WW_lvl) * \
                          (ap.sqrt(self.data2_g) - ap.sqrt(self.data3_g))**2 / \
                          (1.0e-3+self.overlaps)
        error = ap.sum( self.error_map )
        
        # make an overlap mask...
        
        return error

    def speckle_tracking_update(self, steps=8, window=16, search_window=50):
        from multiprocessing import Pool
        import itertools
          
        delta_ij     = self.delta_ij
        delta_ij_sub = np.zeros_like(delta_ij[:, ::steps, ::steps])
        mask         = np.array(self.mask_g).astype(np.bool)
        
        # polymask
        mask_poly = mask.copy()
        w = 60
        mask_poly[:mask.shape[0]//4-w  , :] = True
        mask_poly[3*mask.shape[0]//4+w:, :] = True
        
        mask_poly[:, :mask.shape[1]//4-w  ] = True
        mask_poly[:, 3*mask.shape[1]//4+w:] = True
        
        Os     = []
        deltas = []
        NCCs   = []
        errors = []
        
        errors.append(self.calc_error(delta_ij))
        print 'Error:', errors[-1]

        Os.append(np.array(self.O_g))
        deltas.append(np.array(delta_ij))
        NCCs.append(np.zeros_like(delta_ij[0]))
        
        pool = Pool(processes=self.data2_g.shape[0])
        for ii in range(10):
            print '\n\nloop :', ii
            
            print 'setting up... '
            frames_g   = self.data2_g / (1.0e-3 + self.WW_g.reshape((1, self.W_g.shape[0], self.W_g.shape[1])) )
            forwards_g = self.data3_g/ (1.0e-3 + self.WW_g.reshape((1, self.W_g.shape[0], self.W_g.shape[1])) )
            frames     = np.array(frames_g)
            forwards   = np.array(forwards_g)
            
            #speckle_track_np(forwards[0], frames[0], mask, 6)
            
            print 'sending to workers '
            print forwards.shape, frames.shape, mask.shape, window, search_window, steps
            args = itertools.izip( forwards, frames, itertools.repeat(mask), itertools.repeat(window), \
                                   itertools.repeat(search_window), itertools.repeat(steps) )
            res  = pool.map(speckle_track_np_wrap, args)
            
            print 'workers are done'
            nccs = np.array([i[1][::steps, ::steps] for i in res])
            di   = np.array([i[0][0][::steps, ::steps] for i in res])
            dj   = np.array([i[0][1][::steps, ::steps] for i in res])
            
            print nccs.shape, di.shape, dj.shape, delta_ij_sub.shape
            # do a weigted sum
            norm = np.sum(nccs, axis=0) + 1.0e-10
            delta_ij_sub[0] += np.sum(di*nccs, axis=0) / norm
            delta_ij_sub[1] += np.sum(dj*nccs, axis=0) / norm
            
            # fit displacements to a 2D polynomial
            print 'fitting the pixel shifts to a 2d polynomial'
            coeff_i, fit = polyfit2d(delta_ij_sub[0], mask_poly[::steps, ::steps], 15)
            coeff_j, fit = polyfit2d(delta_ij_sub[1], mask_poly[::steps, ::steps], 15)
            
            # evaluate on finer grid
            i = np.linspace(-1, 1, mask.shape[0])
            j = np.linspace(-1, 1, mask.shape[1])
            delta_ij[0] = P.polygrid2d(i, j, coeff_i)
            delta_ij[1] = P.polygrid2d(i, j, coeff_j)
            
            errors.append(testing.calc_error(delta_ij))
            print 'Error:', errors[-1]
            
            if errors[-1] > errors[-2] and ii > 0 :
                break
            
            Os.append(np.array(testing.O_g))
            deltas.append(np.array(delta_ij))
        
        return delta_ij, Os, errors
    
def speckle_track_np_wrap(x):
    return speckle_track_np(*x)
        

def speckle_track_np(forward, frame, mask, window_size, big_window_size, steps=4):
    from skimage import data
    from skimage.feature import match_template
    # fill the frame with "forward" for masked pixels
    frame[~mask] = forward[~mask]
    
    delta_ij       = np.zeros((2,) + frame.shape, dtype=np.float)
    small_delta_ij = np.zeros_like(delta_ij[:, ::steps, ::steps])

    NCC       = np.zeros(frame.shape, dtype=np.float)
    small_NCC = np.zeros_like(NCC[::steps, ::steps])
    
    # now we will scan through subregions of forward
    # looking for the location of that subregion in 
    # frame
    N               = 2
    #big_window_size = N * window_size
    shape           = forward.shape
    for i in range(0, shape[0], steps):
        if i < (shape[0]//4 - window_size//4) or i >= (3*shape[0]//4 + window_size//4):
            continue
            
        for j in range(0, shape[1], steps):
            if j < (shape[1]//4 - window_size//4) or j >= (3*shape[1]//4 + window_size//4):
                continue
            
            # place the pixel in question at the centre of the window
            # feature in "forward"
            forward_sub = [max(0, i-window_size//2), min(shape[0], i + window_size//2), max(0, j-window_size//2), min(shape[1], j + window_size//2)]
            
            # region in "frame", this is twice the size
            frame_sub = [max(0, i - big_window_size//2), min(shape[0], i + big_window_size//2), \
                         max(0, j - big_window_size//2), min(shape[1], j + big_window_size//2)]
            #frame_sub = [0, shape[0], 0, shape[1]]
            
            # search for a match
            result = match_template(frame[frame_sub[0]:frame_sub[1], frame_sub[2]:frame_sub[3]], \
                                    forward[forward_sub[0]:forward_sub[1], forward_sub[2]:forward_sub[3]])
            
            # normalise the confidence metric, add 1 and divide by 2 so that result: 0-->1
            result    = (result + 1.) / 2.
            result    = result / np.sum(result)
            
            ## HACK !!!!!!!!!!!!!!!!!!!!!!
            # also I have no idea why but result always seems to have a suspiciously large
            # value at exactly the zero displacement value....???
            # fill that pixel with the average of its neighbours
            i0, j0 = (big_window_size-window_size+1)//2, (big_window_size-window_size+1)//2

            rsh = result.shape
            result[i0, j0] = (result[(i0+1) % rsh[0], j0 % rsh[1]] + result[i0 % rsh[0], (j0+1) % rsh[1]] + result[(i0-1) % rsh[0], j0 % rsh[1]] + result[i0 % rsh[0], (j0-1) % rsh[1]]) / 4.
            
            ij = np.unravel_index(np.argmax(result), result.shape)
            #print 'window courner:', i,j,'found at:', ij, 'expected', [window_size/2, window_size/2], 'sub region frame:', frame_sub, 'subregion forward:', forward_sub, 'delta_ij = ', [ij[0] - window_size/2, ij[1] - window_size/2]
            delta_ij[0, i, j] = ij[0] - i0
            delta_ij[1, i, j] = ij[1] - j0
            NCC[i, j]         = result[ij[0], ij[1]]

    if False :
        import skimage.transform
        small_delta_ij = delta_ij[:, ::steps, ::steps]
        small_NCC      = NCC[::steps, ::steps]
        delta_ij[0] = skimage.transform.rescale(small_delta_ij[0], float(steps)) #- steps/2.
        delta_ij[1] = skimage.transform.rescale(small_delta_ij[1], float(steps)) #- steps/2.
        NCC         = skimage.transform.rescale(small_NCC, float(steps))
    return delta_ij, NCC


def show_speckle_track_np(forward, frame, mask, window_size, steps=4):
    from skimage import data
    from skimage.feature import match_template
    # fill the frame with "forward" for masked pixels
    frame[~mask] = forward[~mask]
    
    delta_ij       = np.zeros((2,) + frame.shape, dtype=np.float)
    small_delta_ij = np.zeros_like(delta_ij[:, ::steps, ::steps])

    NCC       = np.zeros(frame.shape, dtype=np.float)
    small_NCC = np.zeros_like(NCC[::steps, ::steps])
    
    # now we will scan through subregions of forward
    # looking for the location of that subregion in 
    # frame
    N               = 2
    big_window_size = N * window_size
    shape           = forward.shape
    for i in range(0, shape[0], steps):
        if i < (shape[0]//4 - window_size//4) or i >= (3*shape[0]//4 + window_size//4):
            continue
            
        for j in range(0, shape[1], steps):
            if j < (shape[1]//4 - window_size//4) or j >= (3*shape[1]//4 + window_size//4):
                continue
            
            # place the pixel in question at the centre of the window
            # feature in "forward"
            forward_sub = [max(0, i-window_size//2), min(shape[0], i + window_size//2), max(0, j-window_size//2), min(shape[1], j + window_size//2)]
            
            # region in "frame", this is twice the size
            frame_sub = [max(0, i - big_window_size//2), min(shape[0], i + big_window_size//2), \
                         max(0, j - big_window_size//2), min(shape[1], j + big_window_size//2)]
            #frame_sub = [0, shape[0], 0, shape[1]]
            
            # search for a match
            result = match_template(frame[frame_sub[0]:frame_sub[1], frame_sub[2]:frame_sub[3]], \
                                    forward[forward_sub[0]:forward_sub[1], forward_sub[2]:forward_sub[3]])
            
            # normalise the confidence metric, add 1 and divide by 2 so that result: 0-->1
            result    = (result + 1.) / 2.
            result    = result / np.sum(result)
            
            ## HACK !!!!!!!!!!!!!!!!!!!!!!
            # also I have no idea why but result always seems to have a suspiciously large
            # value at exactly the zero displacement value....???
            # fill that pixel with the average of its neighbours
            i0, j0 = (big_window_size-window_size+1)//2, (big_window_size-window_size+1)//2

            rsh = result.shape
            result[i0, j0] = (result[(i0+1) % rsh[0], j0 % rsh[1]] + result[i0 % rsh[0], (j0+1) % rsh[1]] + result[(i0-1) % rsh[0], j0 % rsh[1]] + result[i0 % rsh[0], (j0-1) % rsh[1]]) / 4.
            
            ij = np.unravel_index(np.argmax(result), result.shape)
            #print 'window courner:', i,j,'found at:', ij, 'expected', [window_size/2, window_size/2], 'sub region frame:', frame_sub, 'subregion forward:', forward_sub, 'delta_ij = ', [ij[0] - window_size/2, ij[1] - window_size/2]
            delta_ij[0, i, j] = ij[0] - i0
            delta_ij[1, i, j] = ij[1] - j0
            NCC[i, j]         = result[ij[0], ij[1]]
            
            # -------------------------------------
            # Plot the match
            # -------------------------------------
            import matplotlib.pyplot as plt
            image = frame
            coin = forward
            
            x, y = ij[::-1]
            x_im = delta_ij[1, i, j] + j
            y_im = delta_ij[0, i, j] + i
            print 'i, j', i, j, 'found match at:', ij, '(image window reference frame)', 'image (full) reference frame:', (y_im, x_im), 'confidence:', result[ij[0], ij[1]]

            fig = plt.figure(figsize=(8, 3))
            ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
            ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
            ax3 = plt.subplot(1, 3, 3, adjustable='box-forced')

            vmin = [0.8, 1.2]
            ax1.imshow(coin, interpolation='nearest', cmap='Greys_r', vmin=vmin)
            ax1.set_axis_off()
            ax1.set_title('template')
            # highlight template region
            hcoin, wcoin = forward_sub[1]-forward_sub[0], forward_sub[3]-forward_sub[2]
            rect2 = plt.Rectangle((forward_sub[2], forward_sub[0]), wcoin, hcoin, edgecolor='r', facecolor='none')
            ax1.add_patch(rect2)
            
            ax2.imshow(image, interpolation='nearest', cmap='Greys_r', vmin=vmin)
            ax2.set_axis_off()
            ax2.set_title('image')
            # highlight matched region
            rect = plt.Rectangle((x_im, y_im), wcoin, hcoin, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)
            # highlight searched region
            hsearch, wsearch = frame_sub[1]-frame_sub[0], frame_sub[3]-frame_sub[2]
            rect3 = plt.Rectangle((frame_sub[2], frame_sub[0]), wsearch, hsearch, edgecolor='g', facecolor='none')
            ax2.add_patch(rect3)
            
            ax3.imshow(result, interpolation='nearest', cmap='Greys_r')
            ax3.set_axis_off()
            ax3.set_title('`match_template`\nresult')
            # highlight matched region
            ax3.autoscale(False)
            ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

            plt.show()
            

    import skimage.transform
    small_delta_ij = delta_ij[:, ::steps, ::steps]
    small_NCC      = NCC[::steps, ::steps]
    delta_ij[0] = skimage.transform.rescale(small_delta_ij[0], float(steps)) #- steps/2.
    delta_ij[1] = skimage.transform.rescale(small_delta_ij[1], float(steps)) #- steps/2.
    NCC         = skimage.transform.rescale(small_NCC, float(steps))
    return delta_ij, NCC

def polyfit2d(Z, mask, order):
    print 'setting up A and B matrices...'
    A   = []
    mat = np.zeros((order, order), dtype=np.float)
    i = np.linspace(-1, 1, mask.shape[0])
    j = np.linspace(-1, 1, mask.shape[1])
    for ii in range(order):
        for jj in range(order):
            mat.fill(0)
            mat[ii, jj] = 1
            A.append(P.polygrid2d(i, j, mat)[mask])
    
    A = np.array(A).T
    B = Z[mask].flatten()
    
    print 'sending to np.linalg.lstsq ...'
    coeff, r, rank, s = np.linalg.lstsq(A, B)
    print 'residual:', r
    coeff = coeff.reshape((order, order))
    # return the coefficients and the fit
    fit = P.polygrid2d(i, j, coeff)
    return coeff, fit



class Test():
    def __init__(self, data, mask, W, R, O, dx, Zernike_coefficients): 
        """
        """
        dtype = np.float
        self.Zernike_coefficients = Zernike_coefficients
        self.R    = R
        self.R_g  = ap.array(R)
        
        # make the Zernike polynomials in a cartesian polynomial basis
        self.Zernike_polys        = np.array(make_Zernike_polys(len(Zernike_coefficients)))
        
        # define the x-y grid to evaluate the polynomials 
        # evaluate on our grid [2xN, 2xM] where N and M are the frame dims
        # rectangle in circle domain
        shape = (2*data.shape[1], 2*data.shape[2])
        rat = float(shape[0])/float(shape[1])
        x   = np.sqrt(1. / (1. + rat**2))
        y   = rat * x
        dom = [-y, y, -x, x]
        roi = shape
        self.y   = np.linspace(dom[0], dom[1], shape[0])
        self.x   = np.linspace(dom[2], dom[3], shape[1])
        
        # add pixel values
        self.i, self.j = np.indices(self.y.shape + self.x.shape)

        self.mask = np.zeros(shape, dtype=dtype)
        self.mask[:mask.shape[0], :mask.shape[1]] = mask
        self.mask    = np.roll(self.mask, data.shape[1]/2, axis=0)
        self.mask    = np.roll(self.mask, data.shape[2]/2, axis=1)
        self.mask_g  = ap.array(self.mask)
        
        self.W    = np.zeros(shape, dtype=dtype)
        self.W[:W.shape[0], :W.shape[1]] = W * mask
        self.W    = np.roll(self.W, data.shape[1]/2, axis=0)
        self.W    = np.roll(self.W, data.shape[2]/2, axis=1)
        self.W_g  = ap.array(self.W)
        self.WW_g = self.W_g * self.W_g
        
        self.WW_lvl = np.median( (W*W*mask)[W > 0] )

        self.data2   = np.zeros( (data.shape[0],)+shape, dtype=dtype)
        self.data2[:, :data.shape[1], :data.shape[2]] = data * mask
        self.data2   = np.roll(self.data2, data.shape[1]/2, axis=1)
        self.data2   = np.roll(self.data2, data.shape[2]/2, axis=2)
        self.data2_g     = ap.array(self.data2 * self.W)
        self.whitefields = ap.zeros(self.data2_g.shape, dtype=dtype)

        # the regular pixel values
        i, j = np.indices(self.data2.shape[1 :])
         
        # make the object grid
        Oi =  i.max() + np.max(np.abs(R[:, 0]))
        Oj =  j.max() + np.max(np.abs(R[:, 1]))
        
        if O is None :
            O = np.zeros((int(round(Oi)), int(round(Oj))), dtype=dtype)
        self.O_g     = ap.array(O)
        self.norm_g  = ap.zeros(self.O_g.shape, dtype=dtype)
        Oi, Oj       = np.indices(O.shape)
        self.Oi_g, self.Oj_g = ap.array(Oi.ravel().astype(dtype)), ap.array(Oj.ravel().astype(dtype))
    
        # make a bunch of translation matrices for each frame
        self.tfs = []
        for k in range(self.data2_g.shape[0]):
            self.tfs.append(ap.array([[1, 0, self.R[k, 1]], [0, 1, self.R[k, 0]]], dtype=np.float32).d_array)

        self.tfs_2    = np.zeros((self.data2_g.shape[0], 2, 3), dtype=np.float32)
        self.tfs_2[:] = np.array([[1,0,0],[0,1,0]])
        self.tfs_2[:, 0, 2] = -self.R[:, 1]
        self.tfs_2[:, 1, 2] = -self.R[:, 0]
        self.tfs_2    = ap.array(self.tfs_2).d_array


    def data_to_O(self, delta_ij = None):
        if delta_ij is None :
            # update displacements
            delta_ij             = make_pixel_displacements(self.Zernike_coefficients, self.y, self.x, Zernike_polys = self.Zernike_polys)
        
        self.ii, self.jj = self.i + delta_ij[0], self.j + delta_ij[1]
        
        self.ii_g, self.jj_g = ap.array(self.ii.ravel()), ap.array(self.jj.ravel())
        
        # data * W --> undistorted frames
        self.data3_g = ap.array(af.approx2(self.data2_g.d_array, self.jj_g.d_array, self.ii_g.d_array)).reshape(self.data2_g.shape)
        
        # Whitefield --> undistorted frame
        self.W2_g = ap.array(af.approx2(self.WW_g.d_array, self.jj_g.d_array, self.ii_g.d_array)).reshape(self.W_g.shape)
        
        # tiling to object frame
        self.O_g    *= 0
        self.norm_g *= 0

        self.O_g.d_array    = af.transform(self.data3_g[0].d_array, self.tfs[0], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
        self.norm_g.d_array = af.transform(self.W2_g.d_array,       self.tfs[0], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
        
        for k in range(1, self.data3_g.shape[0]):
            self.O_g.d_array    += af.transform(self.data3_g[k].d_array, self.tfs[k], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
            self.norm_g.d_array += af.transform(self.W2_g.d_array, self.tfs[k], odim0=self.O_g.shape[1], odim1=self.O_g.shape[0])
            
        # normalisation by white field
        self.O_g = self.O_g/(1.0e-5+self.norm_g)
        return self.O_g

    def O_to_data(self, delta_ij = None):
        if delta_ij is None :
            # update displacements
            delta_ij             = make_pixel_displacements(self.Zernike_coefficients, self.y, self.x, Zernike_polys = self.Zernike_polys)
        
        # O to undistorted frames
        self.data3_g  = ap.array(af.transform(self.O_g.d_array, self.tfs_2, odim0=self.data3_g.shape[2], odim1=self.data3_g.shape[1])).reshape(self.data2_g.shape)

        # testing
        self.overlaps = ap.array(af.transform(self.norm_g.d_array, self.tfs_2, odim0=self.data3_g.shape[2], odim1=self.data3_g.shape[1])).reshape(self.data2_g.shape)
        
        # undistorted to distorted frames 
        self.ii, self.jj  = self.i - delta_ij[0], self.j - delta_ij[1]
        self.ii_g, self.jj_g = ap.array(self.ii.ravel()), ap.array(self.jj.ravel())
        self.data3_g      = ap.array(af.approx2(self.data3_g.d_array, self.jj_g.d_array, self.ii_g.d_array, off_grid=1.0)).reshape(self.data2_g.shape)
        self.data3_g     *= self.WW_g.reshape((1, self.W_g.shape[0], self.W_g.shape[1]))
        
        self.overlaps     = ap.array(af.approx2(self.overlaps.d_array, self.jj_g.d_array, self.ii_g.d_array, off_grid=1.0)).reshape(self.data2_g.shape)
        return self.data3_g

    def calc_error(self, delta_ij = None):
        if delta_ij is None :
            delta_ij     = make_pixel_displacements(self.Zernike_coefficients, self.y, self.x, Zernike_polys = self.Zernike_polys)
        self.O_g     = self.data_to_O(delta_ij)
        self.data3_g = self.O_to_data(delta_ij)
        
        # sum |sqrt(I) - sqrt(I_forward)|^2
        self.error_map  = self.mask_g.reshape((1, self.mask_g.shape[0], self.mask_g.shape[1])) * \
                          (self.overlaps > 2.5 * self.WW_lvl) * \
                          (ap.sqrt(self.data2_g) - ap.sqrt(self.data3_g))**2 / \
                          (1.0e-3+self.overlaps)
        error = ap.sum( self.error_map )
        
        # make an overlap mask...
        
        return error
    
    def error_sweep(self, Zernike_unit_vec, steps = 10, step_size = 1., centred = True, return_os = False):
        delta0_ij    = make_pixel_displacements(self.Zernike_coefficients, self.y, self.x, Zernike_polys = self.Zernike_polys)
        delta1_ij    = make_pixel_displacements(Zernike_unit_vec, self.y, self.x, Zernike_polys = self.Zernike_polys)
        
        if centred :
            alphas = np.linspace(-(step_size * steps)/2, (step_size * steps)/2, steps)
        else :
            alphas = np.linspace(0., (step_size * steps), steps)
        
        errors = np.zeros(len(alphas))
        Os     = []
        for i, alpha in enumerate(alphas):
            delta_ij  = delta0_ij + alpha * delta1_ij
            errors[i] = self.calc_error(delta_ij)
            print i, alpha, errors[i]
            
            if return_os :
                Os.append(np.array(self.O_g))
        
        self.errors = errors
        if return_os :
            return errors, Os
        else :
            return errors

    def refine_grid_search(self, steps=11, step_size=1., iters=10):
        Z_des = np.zeros_like(self.Zernike_coefficients)
        Os = []

        # first add a small random offset to self.Zernike_coefficients
        # becuase the error space is a bit discontinuous
        #dZ     = np.random.random(self.Zernike_coefficients.shape) - 0.5
        #dZ[:3] = 0
        #dZ[6:] = 0
        #self.Zernike_coefficients += dZ

        error_mins   = []
        errors_tot   = []
        error_mins.append(self.calc_error())
        Os.append(np.array(self.O_g))
        print '\n\nCurrent Zernike coefficients:', self.Zernike_coefficients
        
        for i in range(iters):
            # find the descent direction
            Z_des.fill(0)
            any_change = 0
            for j in range(3, len(Z)):
                print '\n\nOrder:', j
                
                Z_des[j] = 1.
                e_min = 0
                steps_gr      = steps
                step_size_gr  = step_size
                
                # evaluate the errors along the search direction
                errors = self.error_sweep(Z_des, steps = steps_gr, step_size = step_size_gr, centred=True, return_os = False)
                
                # get the minimum
                alphas = np.linspace(-(step_size_gr * steps_gr)/2, (step_size_gr * steps_gr)/2, steps_gr)
                kk     = np.argmin(errors)
                e_min  = errors[kk]
                a_min  = alphas[kk]
                #a_min, e_min = get_min_poly_scalar(alphas, errors)
                
                # check that the best error is better than the last
                if e_min >= (1.0 - 1.0e-10)*error_mins[-1] :
                    print 'no improvement weight, error, best error:', a_min, e_min, error_mins[-1]
                
                else :
                    # assign the new value 
                    self.Zernike_coefficients += a_min * Z_des
                    print 'Current Zernike coefficients:', self.Zernike_coefficients
                     
                    print 'Assigning step, error, delta error:', a_min, e_min, error_mins[-1] - e_min
                    Os.append(np.array(self.data_to_O()))
                    
                    any_change += 1
            
                    error_mins.append(e_min)
                    errors_tot.append(errors)
                
            if any_change < 2 :
                break

        if any_change > 1 :
            print '\n\nMaximum iterations reached. Done!'
        else :
            print '\n\nNo change in error. Done!'
        
        self.Os          = np.array(Os)
        self.error_mins  = error_mins
        self.errors_tot  = errors_tot
        
        print 'Zernike_coefficients:'
        print self.Zernike_coefficients
        print '\nerrors:'
        print self.error_mins
    def refine_steepest(self, steps=11, step_size=1., iters=10, stepsize_zoom=10.):
        Z     = np.zeros_like(self.Zernike_coefficients) 
        Z_des = np.zeros_like(self.Zernike_coefficients)
        Os = []

        # first add a small random offset to self.Zernike_coefficients
        # becuase the error space is a bit discontinuous
        #self.Zernike_coefficients += np.random.random(self.Zernike_coefficients.shape) * 1.0e-2 - 0.5e-2

        error_mins   = []
        errors_tot   = []
        error_grads  = []
        error_mins.append(self.calc_error())
        Os.append(np.array(self.O_g))
        
        for i in range(iters):
            # find the descent direction
            Z_des.fill(0)
            print '\n\nFinding search direction:'
            for j in range(3, len(Z)):
                print '\nOrder:', j
                # define the search direction 
                Z.fill(0)
                Z[j] = 1.
                
                # evaluate the errors along the search direction
                steps_gr      = steps
                step_size_gr  = step_size
                
                e_min = 0
                while e_min > (1+1.0e-3)*error_mins[-1] or e_min == 0 :
                    errors = self.error_sweep(Z, steps = steps_gr, step_size = step_size_gr, return_os = False)
                    alphas = np.linspace(-(step_size_gr * steps_gr)/2, (step_size_gr * steps_gr)/2, steps_gr)
                    
                    e_min = min(errors)
                    
                    if e_min > (1+1.0e-3)*error_mins[-1] :
                        step_size_gr = step_size_gr / stepsize_zoom 
                        print 'No smaller error found reducing stepsize, step:', step_size_gr
                
                if e_min < error_mins[-1] :
                    # calculate the gradient at alpha = 0.0
                    poly     = P.polyfit(alphas, errors, 2)
                    polyder  = P.polyder(poly)
                    Z_des[j] = -P.polyval(0.0, polyder)
                
                error_grads.append(errors)
                print 'Gradient:', Z_des[j]
                
            norm = np.sqrt(np.sum(Z_des**2))
            if norm == 0. :
                break

            # normalise ?
            Z_des = Z_des / norm

            print '\n\nSearch direction:', Z_des
            
            e_min = 0
            steps_gr      = steps
            step_size_gr  = step_size/10.
            while e_min >= error_mins[-1] or e_min == 0 :
                  
                # evaluate the errors along the search direction
                errors = self.error_sweep(Z_des, steps = steps_gr, step_size = step_size_gr, centred=False, return_os = False)
                
                # get the minimum
                alphas = np.linspace(0, (step_size_gr * steps_gr), steps_gr)

                kk = np.argmin(errors)
                e_min = errors[kk]
                a_min = alphas[kk]
                #a_min, e_min = get_min_poly_scalar(alphas, errors)
                
                # check that the best error is better than the last
                if e_min >= error_mins[-1] and e_min > 0. :
                    print '\nno improvement weight, error, best error:', a_min, e_min, error_mins[-1]
                    step_size_gr = step_size_gr / stepsize_zoom 
                    print '\nNo smaller error found reducing stepsize, step:', step_size_gr
                         
            # assign the new value 
            self.Zernike_coefficients += a_min * Z_des
            
            #self.Os.append(np.array(self.O_g))
            print '\nAssigning step, error:', a_min, e_min
            
            error_mins.append(e_min)
            errors_tot.append(errors)
            Os.append(np.array(self.data_to_O()))

        if norm == 0 :
            print '\n\nZero gradient field. Done!'
        else :
            print '\n\nDone!'
        
        self.Os          = np.array(Os)
        self.error_grads = error_grads
        self.error_mins  = error_mins
        self.errors_tot  = errors_tot
        
        print 'Zernike_coefficients:'
        print self.Zernike_coefficients
        print '\nerrors:'
        print self.error_mins
    def make_error_map(self, steps=[5,5,5], step_size=[1.,1.,1.], orders = [3,4,5]):
        half_steps = np.array(step_size) * np.array(steps) / 2.
        i = np.linspace(-half_steps[0], half_steps[0], steps[0])
        j = np.linspace(-half_steps[1], half_steps[1], steps[1])
        k = np.linspace(-half_steps[2], half_steps[2], steps[2])

        if steps[0] == 1 :
            i.fill(0)
        if steps[1] == 1 :
            j.fill(0)
        if steps[2] == 1 :
            k.fill(0)
        error_map = np.zeros(tuple(steps))
        
        for ii in range(error_map.shape[0]):
            for jj in range(error_map.shape[1]):
                self.Zernike_coefficients[orders[0]] = i[ii]
                self.Zernike_coefficients[orders[1]] = j[jj]
                Z_des = np.zeros_like(self.Zernike_coefficients)
                Z_des[orders[2]] = 1.
                error_map[ii, jj, :] = self.error_sweep(Z_des, steps = steps[-1], step_size = step_size[-1], centred=True, return_os = False)
                
                print ii, jj, self.Zernike_coefficients
        return error_map


def make_pixel_displacements(Zernike_coefficients, y, x, Zernike_polys = None):
    Zernike_grad_polys = make_Zernike_gradients(Zernike_coefficients, Zernike_polys)
    
    # evaluate on grid
    grad_grids = (P.polygrid2d(y, x, Zernike_grad_polys[0]), P.polygrid2d(y, x, Zernike_grad_polys[1]))

    return np.array(grad_grids)


def make_Zernike_gradients(Zernike_coefficients, Zernike_polys = None):
    if Zernike_polys is None :
        # make the Zernike polynomials in a cartesian polynomial basis
        Zernike_polys = np.array(make_Zernike_polys(len(Zernike_coefficients)))

    Z = np.dot(Zernike_polys.T, Zernike_coefficients).T
    
    Zernike_grad_polys = (P.polyder(Z, axis=0), P.polyder(Z, axis=1))
    return Zernike_grad_polys

    
        
def make_Zernike_polys(max_Noll_index, orthonormal=False, mask=None):
    # --------------------------------
    # Generate the Zernike polynomials
    # --------------------------------
    Zernike_orders = max_Noll_index
    
    import optics.utils
    # list the Zernike indices in the Noll indexing order:
    # ----------------------------------------------------
    Noll_indices = optics.utils.utils.make_Noll_index_sequence(Zernike_orders)

    if orthonormal :
        if mask is None :
            mask = np.ones((256,256), dtype=np.float)
        
        Z_polynomials = optics.fit_Zernike.make_Zernike_basis(mask, max_order = max_Noll_index)
    else :
        # generate the Zernike polynomials in a cartesian basis:
        # ------------------------------------------------------
        Z_polynomials = []
        for j in range(1, Zernike_orders+1):
            n, m, name           = Noll_indices[j]
            mat, A               = optics.utils.utils.make_Zernike_polynomial_cartesian(n, m, order = Zernike_orders)
            Z_polynomials.append(mat)

    return Z_polynomials


def get_min_poly_scalar(x, errors):
    """
    """
    # fit polynomial, to get over local min
    from numpy.polynomial import polynomial as P
    
    # fit a polynomial of order N / 2
    poly = P.polyfit(x, errors, int(round(len(x)/2.)))
    
    # left and right bounds
    x_l, x_r = min(x), max(x)

    # find the roots of the polynomial
    polyder       = P.polyder(poly)
    polyder_roots = P.polyroots(polyder)
    
    # only accepts roots that are real and fall within the bounds
    polyder_roots = [p for p in polyder_roots if p.imag == 0 and p.real >= x_l and p.real <= x_r]

    # if we actually have a root then check it
    if len(polyder_roots) > 0 :
        # get the x value at the minimum
        x_min         = polyder_roots[np.argmin([P.polyval(r.real, poly) for r in polyder_roots])].real
        
        # get the minimum error 
        e_min         = P.polyval(x_min, poly)
        
        # check that the egdes are not less than the root
        if P.polyval(x_l, poly) < e_min :
            x_min = x_l
            e_min = errors[np.argmin(x)]
        
        elif P.polyval(x_r, poly) < e_min :
            x_min = x_r
            e_min = errors[np.argmax(x)]
        
        # check if this is better than the sampled locations
        i = np.argmin(errors)
        if e_min > errors[i] :
            e_min = errors[i]
            x_min = x[i]
    
    # if no roots were found then just get the smallest error
    else :
        i = np.argmin(errors)
        e_min = errors[i] 
        x_min = x[i]

    return x_min, e_min


class GPU_stitch():
    def __init__(self, data, mask, W, R, O, dx, Zernike_coefficients, grad_grids): 
        # -------------------
        # Stich with gradient
        # -------------------
        # now let's look at the effect of defocus grad_grids[3]
        order = 3
        
        # the regular pixel values
        i, j = np.indices(data.shape[1 :])
         
        # make the object grid
        Oss =  i.max() + np.max(np.abs(R[:, 0]))
        Ofs =  j.max() + np.max(np.abs(R[:, 1]))
        
        if O is None :
            O = np.zeros((int(round(Oss)), int(round(Ofs))), dtype=np.float32)
        Oss, Ofs = np.indices(O.shape)
        
        O_dx = dx
        if O_dx is not None :
            Oss = Oss.astype(np.float) * O_dx[0]
            Ofs = Ofs.astype(np.float) * O_dx[1]
        
        self.error_mins = []
        
        print('sending stuff to the gpu')
        # now stitch send stuff to the gpu 
        dtype = np.float32
        self.dtype = dtype
        self.grad_grids = grad_grids
        self.grad_ss_order_g = ap.array(grad_grids[order][0].ravel().astype(dtype))
        self.grad_fs_order_g = ap.array(grad_grids[order][1].ravel().astype(dtype))
        self.grad_ss_g       = ap.array(np.zeros_like(grad_grids[order][0]).ravel().astype(dtype))
        self.grad_fs_g       = ap.array(np.zeros_like(grad_grids[order][1]).ravel().astype(dtype))
        
        self.i_g       = ap.array(i.ravel().astype(dtype))
        self.j_g       = ap.array(j.ravel().astype(dtype))
        
        self.data_g    = ap.array((mask*data).astype(dtype))
        self.W_g       = ap.array((mask*W).astype(dtype))
        self.mask_g    = ap.array(mask.astype(dtype))
        self.Oss_g     = ap.array(Oss.astype(dtype).ravel())
        self.Ofs_g     = ap.array(Ofs.astype(dtype).ravel())
        self.R_g       = ap.array(R.astype(dtype))
        self.O_g       = ap.array(O.astype(dtype))
        
        self.weights   = np.zeros((len(grad_grids),), dtype=dtype)
        
        if Zernike_coefficients is not None :
            self.weights[: len(Zernike_coefficients)] = \
                    Zernike_coefficients[: min(len(self.weights), len(Zernike_coefficients))]
        self.Os = []
        
        self.orders = len(self.weights)
        
        # object pixels in each frame 
        self.get_i_k_g = lambda k : self.Oss_g + self.R_g[k, 0]
        self.get_j_k_g = lambda k : self.Ofs_g + self.R_g[k, 1]
        
        # undistorted frame values 
        self.get_i_grad_g = lambda w : self.i_g + w*self.grad_ss_order_g + self.grad_ss_g
        self.get_j_grad_g = lambda w : self.j_g + w*self.grad_fs_order_g + self.grad_fs_g
        
    def calculate_grads(self, order):
        self.order = order
        
        # calculate the gradients for this order
        self.grad_ss_order_g = ap.array(self.grad_grids[order][0].ravel().astype(self.dtype))
        self.grad_fs_order_g = ap.array(self.grad_grids[order][1].ravel().astype(self.dtype))
        
        # calculate the gradient for all remaining orders
        grad_y = np.sum([self.weights[o] * self.grad_grids[o][0] for o in range(self.orders) if o != order], axis=0)
        grad_x = np.sum([self.weights[o] * self.grad_grids[o][1] for o in range(self.orders) if o != order], axis=0)
        
        self.grad_ss_g       = ap.array(grad_y.ravel().astype(self.dtype))
        self.grad_fs_g       = ap.array(grad_x.ravel().astype(self.dtype))
        
        self.grad_y_tot = np.sum([self.weights[o] * self.grad_grids[o][0] for o in range(self.orders)], axis=0)
        self.grad_x_tot = np.sum([self.weights[o] * self.grad_grids[o][1] for o in range(self.orders)], axis=0)

    def stitch_gpu(self, weight=None):
        if weight is None :
            self.calculate_grads(0)
            ii_g     = self.i_g + ap.array(self.grad_y_tot.ravel().astype(self.dtype))
            jj_g     = self.j_g + ap.array(self.grad_x_tot.ravel().astype(self.dtype))
        else :
            ii_g     = self.get_i_grad_g(weight)
            jj_g     = self.get_j_grad_g(weight)
        
        self.O_g, self.norm = stitch_gpu(self.data_g, self.W_g, (ii_g, jj_g), (self.get_i_k_g, self.get_j_k_g), self.O_g)
        return self.O_g

    def stitch_gpu_error(self, weight, calculate_stitch=True):
        if calculate_stitch is True :
            self.stitch_gpu(weight)
        
        O = np.array(self.O_g)
        O = O[O>0]
        error = - np.sqrt(np.var(O))# / float(O.size))
        # distorted frame values in the object
        #get_i_grad_forward_g = lambda k : self.i_g - weight*self.grad_ss_order_g - self.R_g[k, 0]
        #get_j_grad_forward_g = lambda k : self.j_g - weight*self.grad_fs_order_g - self.R_g[k, 1]
        #error, frames = stitch_gpu_error(self.data_g, self.W_g, (get_i_grad_forward_g, get_j_grad_forward_g), self.norm, self.O_g, True)
        #self.frames = frames
        return error

    def refine_defocus(self, order = 3):
        import scipy.optimize
        # loop over orders refining as we go
        defocus = np.linspace(-1, 1, 21)
        self.errors = []
        for d in defocus :
            self.calculate_grads(order)
            self.errors.append(self.stitch_gpu_error(d))
            print order, d, self.errors[-1]
            self.Os.append(np.array(self.O_g))

    def refine(self, orders):
        """
        Lots of magic numbers
        """
        # calculate the first O
        self.stitch_gpu(0.0)
        
        # loop over orders refining as we go
        error_mins = [1000. ] #self.stitch_gpu_error(self.weights[3])]
        self.Os.append(np.array(self.O_g))
        errors_tot = []
        pixel_shifts = 10 * 20.
        for j in range(1):
            pixel_shifts = pixel_shifts / 10.
            for order in range(3, orders):
                # pre-calculate the gradients to sweep
                self.calculate_grads(order)
                
                # scale the weights so that 
                # std(grads) == pixel_shifts
                s = np.std(np.array(self.grad_grids[order]))
                bound = pixel_shifts / s
                
                bounds  = [-bound, bound]
                weights = self.weights[order] + np.linspace(bounds[0], bounds[1], 20)
                errors  = np.zeros((len(weights),), np.float)
                
                for i, w in enumerate(weights) :
                    errors[i] = self.stitch_gpu_error(w)
                    print order, w, errors[i]
                    self.Os.append(np.array(self.O_g))

                w_min, e_min = get_min_poly_scalar(weights, errors)
                
                # check that the best error is better than the last
                if e_min < error_mins[-1] :
                    # assign the new value 
                    self.stitch_gpu(w_min)
                    self.weights[order] = w_min
                    #self.Os.append(np.array(self.O_g))
                    
                    print '\nAssigning order, weight, error:', order, w_min, e_min
                    error_mins.append(e_min)
                    errors_tot.append(errors)
                else :
                    print '\nno improvement order, weight, error, best error:', order, w_min, e_min, error_mins[-1]
        self.error_mins = error_mins

def stitch_gpu(data_g, W_g, frame_coords_g, global_coords_func, O_g): 
    O2_g = ap.zeros((O_g.shape[0] * O_g.shape[1],), dtype=O_g.dtype)
    norm = ap.zeros((O_g.shape[0] * O_g.shape[1],), dtype=O_g.dtype)
    
    ii_g = frame_coords_g[0].d_array
    jj_g = frame_coords_g[1].d_array
    
    # un-distort the whitefield**2
    WW_cor_g = ap.array(af.approx2((W_g*W_g).d_array, jj_g, ii_g)).reshape(W_g.shape)
    
    for k in range(data_g.shape[0]):
        # un-distort frame
        data_cor_g = ap.array(af.approx2((W_g*data_g[k]).d_array, jj_g, ii_g)).reshape(W_g.shape)
        
        # get the coords in the object reference frame
        O_i_g = global_coords_func[0](k).d_array
        O_j_g = global_coords_func[1](k).d_array
        
        # add un-distorted frame to global data*whitefield array
        O2_g.d_array  += af.approx2( data_cor_g.d_array, O_j_g, O_i_g)
        
        # add frame to global whitefield**2 array
        norm += af.approx2( (WW_cor_g).d_array, O_j_g, O_i_g)
    
    norm[norm<1.0e-5] = 1.
    O_g = (O2_g / norm).reshape(O_g.shape)
    return O_g, norm

def stitch_gpu_error(data_g, W_g, forward_coords_func, norm_g, O_g, return_frames=False):
    """
    compare data/whitfield <--> O_g[k]
    """
    W2_g = ap.array(W_g) 
    W2_g[W2_g<1.0e-5] = 1.
    
    error_tot = 0.0
    norm_g = norm_g.reshape(O_g.shape)
    frames = []

    #print '\n\n\n\n\n'
    for k in range(data_g.shape[0]):
        i = forward_coords_func[0](k).d_array
        j = forward_coords_func[1](k).d_array
        
        # distorted frame
        #print ap.array(i)[0], ap.array(j)[0]
        d_g = W_g*ap.array(af.approx2(O_g.d_array , j, i)).reshape(W_g.shape)
        n_g = ap.array(af.approx2(norm_g.d_array, j, i)).reshape(W_g.shape)
        
        # calculate the error (in future: weighted by the overlap)
        mean = ap.mean(n_g)
        m_g  = n_g > 0.2 * mean
        n_g[n_g<1.0e-5] = 1.
        
        #error = ap.sum( m_g*(d_g - data_g[k])**2 / n_g)
        errs = np.array(m_g*(d_g - data_g[k])**2 / n_g)
        median, std = np.std(errs), np.median(errs)
        errs = errs[np.abs(errs - median) < 4*std]
        error = np.sum( errs )
        #error = ap.sum( m_g*(d_g - data_g[k]/W2_g)**2 / n_g)
        #print('frame, error:', k, error)
        
        if return_frames :
            frames.append(np.array(d_g))
        error_tot += error
    if return_frames :
        return error_tot, frames
    else :
        return error_tot


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
    Oss =  i.max() + np.max(np.abs(R[:, 0]))
    Ofs =  j.max() + np.max(np.abs(R[:, 1]))
    
    if O is None :
        O = np.zeros((int(round(Oss)), int(round(Ofs))), dtype=np.float)
    Oss, Ofs = np.indices(O.shape)
    
    if O_dx is not None :
        Oss = Oss.astype(np.float) * O_dx[0]
        Ofs = Ofs.astype(np.float) * O_dx[1]

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


def make_O_grad_gpu(data, R, W, mask, grad_ss_fs, Oss_fs, O=None):
    """
    These must all be afnumpy arrays:
    
    data : [K, N, M] array of projection images
    R    : [K, 3]    array of pixel shifts for each frame
    W    : [N*M]     raveled whitefield array
    mask : [N*M]     raveled mask
    grad_ss_fs : (grad_ss, grad_fs) pixel values for the data[k]
        grad_ss : raveled pixel coords + offsets [N*M] along the slow scan direction
        grad_fs : raveled pixel coords + offsets [N*M] along the fast scan direction
    Oss_fs : the slow and fast scan coordinates for O in pixel units, need
             not have increments of 1.
    
    if you already have memory allocated for O then give it to me (saves time)
    """
    get_i_k_O = lambda k : Oss_fs[0] + R[k, 0]
    get_j_k_O = lambda k : Oss_fs[1] + R[k, 1]
    
    if O is None :
        O  = ap.zeros(O_ss_fs.shape, dtype=np.float).ravel()
    
    norm   = ap.zeros(O_g.shape, O_g.dtype)
    data_i = ap.zeros(data[0].shape, data[0].dtype)
    
    # un-distort the whitefield
    W_cor = af.approx2( (W*W).d_array, grad_ss_fs[1].d_array, grad_ss_fs[0].d_array)
    W_cor = W_cor.reshape(W.shape)
    
    for k in range(data.shape[0]):
        # un-distort the data
        d_cor = af.approx2( (W*data[k]).d_array, grad_ss_fs[1].d_array, grad_ss_fs[0].d_array)
        d_cor = d_cor.reshape(W.shape)
        
        O    += af.approx2( (d_cor).d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        norm += af.approx2( (W_cor).d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)


def fit_Zernike_grads(data, R, W, O, mask, Z_grads, Z_coef, fit_grads=False, orders=36, O_dx = None, iters=1):
    """
    Z_grads = [(dss, dfs), ...], (len = oders)
    """
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
    
    print('sending stuff to the gpu')
    # now stitch send stuff to the gpu 
    data_g    = ap.array((mask*data).astype(np.float))
    W_g       = ap.array((mask*W).astype(np.float))
    Oss_g     = ap.array(Oss.astype(np.float).ravel())
    Ofs_g     = ap.array(Ofs.astype(np.float).ravel())
    R_g       = ap.array(R.astype(np.float))
    O_g       = ap.array(O.astype(np.float).ravel())
    
    # to refine we need a cost function

    for order in range(orders):
        pass
    
    get_i_k_g = lambda k : Oss_g + dss + R[k, 0]
    get_j_k_g = lambda k : Ofs_g + dfs + R[k, 1]

    return O, P, Z_out, errors

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
    
    #if params['gpu_stitch']['o_dx'] is not None :
    #    dx = params['gpu_stitch']['o_dx'] / du

    # if the shape but not the spacing is given then scale
    #elif params['gpu_stitch']['o_shape'] is not None :
    #    Y     = (data.shape[0]-1 + np.max(np.abs(R[:, 0]))) 
    #    X     = (data.shape[1]-1 + np.max(np.abs(R[:, 1])))
    #    shape = params['gpu_stitch']['o_shape']
    #    dx    = [Y/float(shape[0]), Y/float(shape[1])]
    #else :
    #    dx = [1., 1.]
    
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

    # Zernike polynomials
    # -------------------
    # get the list of zernike polynomial coefficients 
    # if there are any
    #if params['gpu_stitch']['zernike'] is not None :
    #    print params['gpu_stitch']['zernike']
    #    Z = f[params['gpu_stitch']['zernike']][()].astype(np.float)
    #else :
    #    Z = np.zeros( (params['gpu_stitch']['orders'],), dtype=np.float)

    #fit_grads = params['gpu_stitch']['fit_grads']
    #orders    = params['gpu_stitch']['orders']
    
    #test = Test(data, mask, W, R, O, dx, Z) 

    """
    forward = np.array(testing.data3_g[31])
    frame   = np.array(frames_g[31])
    d, NCC = speckle_track_np(forward, frame, mask, 6)
    """
    
    testing      = Test2(data, mask, W, R, None, delta_ij) 
    
    if params['gpu_stitch']['fit_grads'] :
        delta_ij, Os, errors = testing.speckle_tracking_update(steps=params['gpu_stitch']['steps'], \
                                                               window=params['gpu_stitch']['window'], \
                                                               search_window=params['gpu_stitch']['search_window'])
    else :
        print 'stitching...'
        Os       = [np.array(testing.data_to_O_min_mem(testing.delta_ij))]
        delta_ij = testing.delta_ij
        errors   = [0]

    print 'Object Field of view:', np.array(Os[-1].shape) * du
    print 'Object shape:        ', Os[-1].shape
    print 'Pixel size:          ', du
    """

    testing      = Test2(data, mask, W, R, O) 
    delta_ij     = np.array([np.zeros_like(testing.i), np.zeros_like(testing.j)]).astype(np.float)
    delta_ij_sub = np.zeros_like(delta_ij[:, ::steps, ::steps])
    mask         = np.array(testing.mask_g).astype(np.bool)

    # polymask
    mask_poly = mask.copy()
    w = 60
    mask_poly[:mask.shape[0]//4-w  , :] = True
    mask_poly[3*mask.shape[0]//4+w:, :] = True
    
    mask_poly[:, :mask.shape[1]//4-w  ] = True
    mask_poly[:, 3*mask.shape[1]//4+w:] = True

    Os     = []
    deltas = []
    NCCs   = []
    errors = []

    errors.append(testing.calc_error(delta_ij))
    print 'Error:', errors[-1]

    Os.append(np.array(testing.O_g))
    deltas.append(np.array(delta_ij))
    NCCs.append(np.zeros_like(delta_ij[0]))
    
    def speckle_track_np_wrap(x):
        return speckle_track_np(*x)
    
    pool = Pool(processes=data.shape[0])
    for ii in range(20):
        print '\n\nloop :', ii
        
        print 'setting up... '
        frames_g   = testing.data2_g / (1.0e-3 + testing.WW_g.reshape((1, testing.W_g.shape[0], testing.W_g.shape[1])) )
        forwards_g = testing.data3_g/ (1.0e-3 + testing.WW_g.reshape((1, testing.W_g.shape[0], testing.W_g.shape[1])) )
        frames     = np.array(frames_g)
        forwards   = np.array(forwards_g)

        #speckle_track_np(forwards[0], frames[0], mask, 6)
        
        print 'sending to workers '
        args = itertools.izip( forwards, frames, itertools.repeat(mask), itertools.repeat(window), \
                               itertools.repeat(search_window), itertools.repeat(steps) )
        res  = pool.map(speckle_track_np_wrap, args)

        print res[0][0].shape, res[0][1].shape
        print 'workers are done'
        nccs = np.array([i[1][::steps, ::steps] for i in res])
        di   = np.array([i[0][0][::steps, ::steps] for i in res])
        dj   = np.array([i[0][1][::steps, ::steps] for i in res])
        
        print nccs.shape, di.shape, dj.shape, delta_ij_sub.shape
        # do a weigted sum
        norm = np.sum(nccs, axis=0) + 1.0e-10
        delta_ij_sub[0] += np.sum(di*nccs, axis=0) / norm
        delta_ij_sub[1] += np.sum(dj*nccs, axis=0) / norm
        
        # fit displacements to a 2D polynomial
        print 'fitting the pixel shifts to a 2d polynomial'
        coeff_i, fit = polyfit2d(delta_ij_sub[0], mask_poly[::steps, ::steps], 15)
        coeff_j, fit = polyfit2d(delta_ij_sub[1], mask_poly[::steps, ::steps], 15)
        
        # evaluate on finer grid
        i = np.linspace(-1, 1, mask.shape[0])
        j = np.linspace(-1, 1, mask.shape[1])
        delta_ij[0] = P.polygrid2d(i, j, coeff_i)
        delta_ij[1] = P.polygrid2d(i, j, coeff_j)
        
        errors.append(testing.calc_error(delta_ij))
        print 'Error:', errors[-1]

        if errors[-1] > errors[-2] and ii > 0 :
            break
        
        Os.append(np.array(testing.O_g))
        deltas.append(np.array(delta_ij))
    
    #print 'setting up... '
    #frames_g = testing.data2_g / (1.0e-3 + testing.WW_g.reshape((1, testing.W_g.shape[0], testing.W_g.shape[1])) )
    #forwards = np.array(testing.data3_g)
    #frames   = np.array(frames_g)

    #d, n = show_speckle_track_np(forwards[31], frames[31], mask, 20, steps=40)
    """

    """
    import matplotlib.pyplot as plt
    result = test.result
    ij = test.ij
    coin = test.forward_sub_im
    image = test.frame_sub_im
    
    x, y = ij[::-1]

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 3))

    ax1.imshow(coin)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(image)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    hcoin, wcoin = coin.shape
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()
    """


    """
    # rectangle in circle domain
    rat = float(ROI[1]-ROI[0])/float(ROI[3]-ROI[2])
    x   = np.sqrt(1. / (1. + rat**2))
    y   = rat * x
    dom = [-y, y, -x, x]
    
    # Get the Zernike gradients
    grads, grad_grids, basis, basis_grids = optics.fit_Zernike.make_Zernike_grads(mask, roi = [0, data.shape[1], 0, data.shape[2]], max_order = orders, \
                                   return_grids = True, return_basis = True, yx_bounds = dom)
    
    #####################
    # Refine O and W
    #####################
    gpu_stitcher = GPU_stitch(data, mask, W, R, O, dx, Z, grad_grids)
    
    if fit_grads :
        gpu_stitcher.refine(orders)
        O = np.array(gpu_stitcher.O_g)
    else :
        O = np.array(gpu_stitcher.stitch_gpu())
    P = W
    
    # get the phase :
    E  = f['/entry_1/instrument_1/source_1/energy'][()]
    wavelen = sc.h * sc.c / E
    z  = f['/entry_1/instrument_1/detector_1/distance'][()]
    du2 = [f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/y_pixel_size'][()]]

    # get the x-y grid
    xscale = data.shape[2] * du[1] / (dom[3]-dom[2])
    x = np.linspace(dom[2]*xscale, dom[3]*xscale, data.shape[2])

    yscale = data.shape[1] * du[0] / (dom[1]-dom[0])
    y = np.linspace(dom[0]*yscale, dom[1]*yscale, data.shape[1])
    
    c = 2.0*np.pi * du2[0] / (wavelen * z)
    
    from numpy.polynomial import polynomial as PP
    phase = c * np.sum([gpu_stitcher.weights[o] * c * PP.polygrid2d(y, x, basis[o]) for o in range(gpu_stitcher.orders)], axis=0)

    if params['gpu_stitch']['normalise'] :
        a = np.mean(np.abs(O))
        P *= a
        O /= a
    
    W = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    W[ROI[0]:ROI[1], ROI[2]:ROI[3]] = P[:].real
    """
    
    # write the result 
    ##################
    if params['gpu_stitch']['output_file'] is not None :
        g = h5py.File(params['gpu_stitch']['output_file'])
        outputdir = os.path.split(params['gpu_stitch']['output_file'])[0]
    else :
        g = f
        outputdir = os.path.split(args.filename)[0]
    
    # Zernike coefficients
    #if fit_grads :
    #    key = params['gpu_stitch']['h5_group']+'/Zernike_coefficients'
    #    if key in g :
    #        del g[key]
    #    g[key] = gpu_stitcher.weights

    # phase
    #if fit_grads :
    #    key = params['gpu_stitch']['h5_group']+'/phase_gpu_stitch'
    #    if key in g :
    #        del g[key]
    #    g[key] = phase

    # errors
    if len(errors) > 1 :
        key = params['gpu_stitch']['h5_group']+'/errors_gpu_stitch'
        if key in g :
            del g[key]
        g[key] = np.array(errors)

    # object history
    if len(Os) > 1:
        key = params['gpu_stitch']['h5_group']+'/Os_gpu_stitch'
        if key in g :
            del g[key]
        g[key] = np.array(Os)
    
    # pixel shifts
    key = params['gpu_stitch']['h5_group']+'/pixel_shifts_fs_gpu_stitch'
    if key in g :
        del g[key]
    grad_x = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    shape = delta_ij[0].shape
    grad_x[ROI[0]:ROI[1], ROI[2]:ROI[3]] = delta_ij[1][shape[0]//4 : 3*shape[0]//4, shape[1]//4 : 3*shape[1]//4]
    g[key] = grad_x
    
    key = params['gpu_stitch']['h5_group']+'/pixel_shifts_ss_gpu_stitch'
    if key in g :
        del g[key]
    grad_y = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
    grad_y[ROI[0]:ROI[1], ROI[2]:ROI[3]] = delta_ij[0][shape[0]//4 : 3*shape[0]//4, shape[1]//4 : 3*shape[1]//4]
    g[key] = grad_y

    # object
    key = params['gpu_stitch']['h5_group']+'/O_gpu_stitch'
    if key in g :
        del g[key]
    g[key] = Os[-1] #np.sqrt(O).astype(np.complex128)
    
    # whitefield
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
